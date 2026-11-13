# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFILL = ROOT / "hcu_sglang/disagg/prefill.py"
INFLIGHT_FN = "process_disagg_prefill_inflight_queue"


def load():
    text = PREFILL.read_text(encoding="utf-8")
    return text, text.splitlines(keepends=True), ast.parse(text)


def function_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def source_of(lines, node):
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def call_attr(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def positional_text(call, index):
    return ast.unparse(call.args[index]) if len(call.args) > index else None


def calls_named(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and call_attr(n) == name]


def keyword_text(lines, call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.unparse(keyword.value)
    return None


class PDBPrefillInflightConsensusContractTests(unittest.TestCase):
    def setUp(self):
        self.text, self.lines, self.tree = load()

    def test_all_reduce_polls_is_imported_from_owned_disagg_utils(self):
        modules = [
            node.module
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "_all_reduce_polls" for alias in node.names)
        ]
        self.assertEqual(modules, ["hcu_sglang.disagg.utils"])

    def test_rank_consensus_checker_comes_from_upstream_utils(self):
        modules = [
            node.module
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "rank_consensus_checker" for alias in node.names)
        ]
        self.assertEqual(modules, ["sglang.srt.utils"])

    def test_inflight_removal_is_collective_across_tp_and_cp(self):
        fn = function_named(self.tree, INFLIGHT_FN)
        reductions = calls_named(fn, "_all_reduce_polls")
        self.assertEqual(len(reductions), 2)
        groups = sorted(positional_text(call, 1) for call in reductions)
        self.assertEqual(groups, ["self.attn_cp_cpu_group", "self.attn_tp_cpu_group"])

    def test_reductions_run_after_poll_loop_and_survive_the_empty_queue_return(self):
        fn = function_named(self.tree, INFLIGHT_FN)
        reductions = calls_named(fn, "_all_reduce_polls")
        poll_loop = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and call_attr(node.iter) == "zip"
        ]
        self.assertEqual(len(poll_loop), 1)
        loop_end = max(node.end_lineno for node in poll_loop[0].body)
        for call in reductions:
            self.assertGreater(call.lineno, loop_end)

    def test_done_reqs_are_rebuilt_from_the_reduced_flags(self):
        fn = function_named(self.tree, INFLIGHT_FN)
        assignments = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "done_reqs" for t in node.targets)
        ]
        self.assertGreaterEqual(len(assignments), 2)
        self.assertIn(
            "keep_ids",
            [
                node.id
                for node in ast.walk(fn)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            ],
        )
        self.assertIn("self.disagg_prefill_inflight_queue = undone_reqs", self.text)

    def test_completed_external_kv_requests_leave_the_queue_before_polling(self):
        fn = function_named(self.tree, INFLIGHT_FN)
        guarded = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
                continue
            names = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
            if names == {"external_kv_cleanup_done", "external_kv_response_sent"}:
                guarded.append(node)
        self.assertEqual(len(guarded), 1)
        body = source_of(self.lines, guarded[0])
        self.assertIn("done_reqs.append(req)", body)
        self.assertIn("continue", body)

    def test_queue_membership_does_not_depend_on_rank_local_cleanup_state(self):
        for fn_name in (INFLIGHT_FN, "handle_bootstrap_failure"):
            with self.subTest(function=fn_name):
                fn = function_named(self.tree, fn_name)
                source = source_of(self.lines, fn)
                self.assertNotIn("if not cleanup_done and req not in", source)
        batch_fn = function_named(self.tree, "process_batch_result_disagg_prefill")
        batch_source = source_of(self.lines, batch_fn)
        self.assertNotIn("elif req not in self.disagg_prefill_inflight_queue", batch_source)

    def test_aborted_fake_transfer_senders_are_aborted_before_queueing(self):
        fn = function_named(self.tree, "process_batch_result_disagg_prefill")
        sender_aborts = [
            call
            for call in calls_named(fn, "abort")
            if ast.unparse(call.func).endswith(".disagg_kv_sender.abort")
        ]
        self.assertEqual(len(sender_aborts), 1)
        guard = source_of(self.lines, fn)
        self.assertLess(
            guard.index("req.bootstrap_host == FAKE_BOOTSTRAP_HOST"),
            guard.index("req.disagg_kv_sender.abort()"),
        )

    def test_stuck_cleanup_warning_is_rate_limited(self):
        fn = function_named(self.tree, INFLIGHT_FN)
        self.assertIn("_inflight_held_since", self.text)
        rearm = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Subscript) for t in node.targets)
        ]
        self.assertTrue(rearm)
        self.assertIn("logger.warning", source_of(self.lines, fn))
        self.assertIn("> 60", source_of(self.lines, fn))


if __name__ == "__main__":
    unittest.main()
