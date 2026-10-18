"""Persistence contract tests for the bounded calibration history."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

from calibration import SCHEMA_VERSION, CalibrationStore


class CalibrationStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use ordinary inherited ACLs and keep fixtures in the ignored runtime
        # area. The sandbox can deny cleanup of private TemporaryDirectory ACLs.
        cls.fixture_root = Path(__file__).resolve().parents[1] / ".runtime" / "calibration-store-tests"
        cls.fixture_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.test_dir = self.fixture_root / uuid.uuid4().hex
        self.test_dir.mkdir()
        self.path = self.test_dir / "calibrations.json"

    @staticmethod
    def report(fingerprint: str, value: int) -> dict:
        return {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "value": value}

    def read_store(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_save_writes_versioned_json_and_replaces_old_file_atomically(self):
        store = CalibrationStore(self.path)
        store.save(self.report("one", 1))
        first_bytes = self.path.read_bytes()
        store.save(self.report("two", 2))

        saved = self.read_store()
        self.assertEqual(saved["schema_version"], SCHEMA_VERSION)
        self.assertEqual(saved["entries"], [self.report("one", 1), self.report("two", 2)])
        self.assertNotEqual(self.path.read_bytes(), first_bytes)
        self.assertEqual(list(self.test_dir.glob("*.tmp")), [])

    def test_corrupt_json_is_discarded_before_saving_new_report(self):
        self.path.write_text('{"schema_version":', encoding="utf-8")

        CalibrationStore(self.path).save(self.report("fresh", 1))

        self.assertEqual(self.read_store(), {
            "schema_version": SCHEMA_VERSION,
            "entries": [self.report("fresh", 1)],
        })

    def test_unknown_schema_is_discarded_before_saving_new_report(self):
        self.path.write_text(json.dumps({"schema_version": 999, "entries": [self.report("old", 0)]}), encoding="utf-8")

        CalibrationStore(self.path).save(self.report("fresh", 1))

        self.assertEqual(self.read_store()["entries"], [self.report("fresh", 1)])

    def test_malformed_entry_list_is_discarded(self):
        self.path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "entries": {"bad": "shape"}}), encoding="utf-8")

        CalibrationStore(self.path).save(self.report("fresh", 1))

        self.assertEqual(self.read_store()["entries"], [self.report("fresh", 1)])

    def test_nonfinite_value_in_existing_json_is_discarded_as_corrupt(self):
        # Python's permissive JSON decoder accepts NaN by default, but the
        # persisted schema and writer use strict JSON and must recover safely.
        self.path.write_text(
            '{"schema_version": 1, "entries": [{"fingerprint": "old", "measurement": NaN}]}',
            encoding="utf-8",
        )

        CalibrationStore(self.path).save(self.report("fresh", 1))

        self.assertEqual(self.read_store()["entries"], [self.report("fresh", 1)])

    def test_entry_count_is_bounded_to_newest_reports(self):
        store = CalibrationStore(self.path, max_entries=2)
        for index in range(4):
            store.save(self.report(f"fingerprint-{index}", index))

        self.assertEqual(self.read_store()["entries"], [
            self.report("fingerprint-2", 2), self.report("fingerprint-3", 3),
        ])

    def test_zero_entry_limit_is_rejected_instead_of_disabling_the_bound(self):
        with self.assertRaises(ValueError):
            CalibrationStore(self.path, max_entries=0)

    def test_saving_same_fingerprint_overwrites_previous_measurement(self):
        store = CalibrationStore(self.path)
        store.save(self.report("same", 1))
        store.save(self.report("other", 7))
        store.save(self.report("same", 2))

        entries = self.read_store()["entries"]
        self.assertEqual(entries, [self.report("other", 7), self.report("same", 2)])
        self.assertEqual(sum(entry["fingerprint"] == "same" for entry in entries), 1)

    def test_failed_atomic_replace_preserves_previous_store(self):
        store = CalibrationStore(self.path)
        original = self.report("good", 1)
        store.save(original)
        original_bytes = self.path.read_bytes()

        with patch("calibration.os.replace", side_effect=OSError("simulated replace failure")):
            with self.assertRaises(OSError):
                store.save(self.report("new", 2))

        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(list(self.test_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
