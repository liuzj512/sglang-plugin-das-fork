# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import unittest

from tools.build_migration_manifest import node_hash


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTriageTests(unittest.TestCase):
    def test_validation_only_sources_preserve_full_ast_and_callers(self):
        manifest = json.loads(
            (ROOT / "tests/contracts/runtime_validation_only_sources.json").read_text()
        )
        self.assertEqual(len(manifest["files"]), 30)
        for row in manifest["files"]:
            with self.subTest(source=row["source"]):
                tree = ast.parse((ROOT / row["owned"]).read_text())
                digest = node_hash(tree)
                self.assertEqual(digest, row["source_ast_sha256"])
                self.assertEqual(digest, row["owned_ast_sha256"])
                self.assertTrue(row["callers"])
                for caller in row["callers"]:
                    self.assertIn(row["owned"][:-3].replace("/", "."), (ROOT / caller).read_text())

    def test_every_changed_definition_has_the_recorded_owned_ast(self):
        manifest = json.loads(
            (ROOT / "tests/contracts/runtime_reviewed_definitions.json").read_text()
        )
        self.assertEqual(len(manifest["files"]), 9)
        for row in manifest["files"]:
            owned = ast.parse((ROOT / row["owned"]).read_text())
            definitions = {}
            for node in ast.walk(owned):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions.setdefault(node.name, []).append(node)
            for item in row["definitions"]:
                with self.subTest(source=row["source"], definition=item["source"]):
                    candidates = definitions[item["owned"].rsplit(".", 1)[-1]]
                    digests = {
                        node_hash(node)
                        for node in candidates
                    }
                    self.assertIn(item["ast_sha256"], digests)

    def test_ast_hash_ignores_only_empty_type_parameter_metadata(self):
        tree = ast.parse("def example(value):\n    return value + 1\n")
        node = tree.body[0]
        fields = node._fields
        if "type_params" not in fields:
            node._fields = (*fields, "type_params")
        node.type_params = []
        expected = node_hash(tree)
        self.assertIn("type_params", node._fields)
        self.assertEqual(node.type_params, [])
        node.type_params = [ast.Name(id="T", ctx=ast.Load())]
        self.assertNotEqual(node_hash(tree), expected)
        node.type_params = []
        changed = ast.parse("def example(value):\n    return value + 2\n")
        self.assertNotEqual(node_hash(changed), expected)

    def test_ast_hash_matches_the_legacy_non_generic_schema(self):
        node = ast.parse("def example(value):\n    return value + 1\n")
        for child in ast.walk(node):
            if getattr(child, "type_params", None) == []:
                child._fields = tuple(field for field in child._fields if field != "type_params")
        expected = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(node_hash(ast.parse("def example(value):\n    return value + 1\n")), expected)

    def test_triage_has_1464_unique_rows_and_explicit_exclusions(self):
        with (ROOT / "plans/hcu-runtime-triage-20261105.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1464)
        self.assertEqual(len({row["path"] for row in rows}), 1464)
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
            if row["category"] == "not_applicable":
                self.assertTrue(row["reason"])
                self.assertEqual(row["verification_lane"], "none")
        self.assertEqual(
            counts,
            {
                "existing_needs_validation": 111,
                "missing_or_partial_implementation": 772,
                "not_applicable": 581,
            },
        )


    def test_configuration_scope_rows_are_marked_with_the_exact_phrase(self):
        decisions = json.loads(
            (ROOT / "plans/sglang-das-decisions.json").read_text()
        )
        phrase = "配置范围外，非等价证明"
        marked = {path: row for path, row in decisions.items() if row.get("review_reason") == phrase}
        self.assertEqual(len(marked), 602)
        models_marked = [path for path in marked if path.startswith("python/sglang/srt/models/")]
        self.assertEqual(len(models_marked), 114)
        self.assertFalse(any(path.endswith("/models/qwen3_5.py") for path in marked))
        with (ROOT / "plans/hcu-runtime-target-closure-20261105.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        from_closure = [row for row in rows if phrase in row["rationale"]]
        self.assertEqual(len(from_closure), 602)
        self.assertEqual(len([row for row in from_closure if row["group"] == "models"]), 112)

if __name__ == "__main__":
    unittest.main()
