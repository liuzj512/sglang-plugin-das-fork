# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]

SOURCE = Path("../input-code/sglang-das/python/sglang/kernels/ops/attention/fla/l2norm.py")
OWNED = ROOT / "hcu_sglang/layers/attention/fla/l2norm_source.py"


class L2NormMigrationTests(unittest.TestCase):
    def test_owned_copy_is_byte_identical_to_source(self):
        if not SOURCE.is_file():
            self.skipTest("sglang-das source tree is not next to the plugin checkout")
        self.assertEqual(
            hashlib.sha256(OWNED.read_bytes()).hexdigest(),
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        )

    def test_vendor_fast_path_is_present_with_triton_fallback(self):
        source = OWNED.read_text()
        self.assertIn("_has_l2norm_cuda", source)
        self.assertIn("torch.ops.sgl_kernel", source)
        self.assertIn("def l2norm_fwd_kernel(", source)
        tree = ast.parse(source)
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "l2norm_fwd"
        )
        guard = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If) and "_has_l2norm_cuda" in ast.unparse(node.test)
        )
        self.assertIn("_l2norm_cuda(", ast.unparse(guard))

    def test_registered_targets_are_the_declared_ones(self):
        contract = json.loads((ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text())
        declared = set(contract["expected_hook_targets"])
        captured = []

        def record(target, hook, hook_type):
            captured.append(target)

        from hcu_sglang.layers.attention.fla import l2norm as hook_module
        from sglang.srt.plugins.hook_registry import HookRegistry

        with patch.object(HookRegistry, "register", side_effect=record):
            hook_module.register()
        self.assertEqual(len(captured), 5)
        for target in captured:
            self.assertIn(target, declared)

    def test_non_hcu_delegates_to_original(self):
        from hcu_sglang.layers.attention.fla import l2norm as hook_module

        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return "original"

        with patch.object(hook_module, "is_hcu", return_value=False), patch.object(
            hook_module, "get_original_attr", return_value=original
        ):
            replacement = hook_module._make_replacement(
                "sglang.kernels.ops.attention.fla.l2norm", "l2norm_fwd"
            )
            self.assertEqual(replacement("x", eps=1e-6), "original")
        self.assertEqual(calls, [(("x",), {"eps": 1e-6})])

    def test_vendor_branch_calls_the_cuda_operator_and_restores_shape(self):
        loader = importlib.util.spec_from_file_location("l2norm_vendor_probe", OWNED)
        module = importlib.util.module_from_spec(loader)

        class FakeTensor:
            shape = (4, 8)
            dtype = torch.bfloat16
            is_cuda = True

            def __init__(self):
                self.views = []

            def contiguous(self):
                return self

            def view(self, *shape):
                self.views.append(shape)
                return self

            def to(self, dtype):
                raise AssertionError("dtype conversion is not expected for bf16 input")

        tensor = FakeTensor()
        seen = []

        def fake_op(x, eps):
            seen.append((x, eps))
            return x

        vendor = ModuleType('sgl_kernel')
        vendor.l2norm = fake_op
        with patch.dict(sys.modules, {'sgl_kernel': vendor}):
            loader.loader.exec_module(module)
        self.assertTrue(module._has_l2norm_cuda)
        self.assertIs(module._l2norm_cuda, fake_op)
        with patch.object(module, "_l2norm_cuda", fake_op):
            result = module.l2norm_fwd(tensor, 1e-6)

        self.assertIs(result, tensor)
        self.assertEqual(seen[0][1], 1e-6)
        self.assertEqual(tensor.views, [(-1, 8), ((4, 8),)])


if __name__ == "__main__":
    unittest.main()
