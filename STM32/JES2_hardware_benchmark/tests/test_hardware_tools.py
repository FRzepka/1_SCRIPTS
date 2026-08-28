from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


serial_benchmark = load_script("collect_serial_benchmark")
memory_report = load_script("extract_memory_report")


class SerialBenchmarkTests(unittest.TestCase):
    def test_parse_valid_result(self):
        result = serial_benchmark.parse_result("RESULT,17,DD,0.6125,48000,OK", "17", "DD")
        self.assertEqual(result["cycles"], 48000)
        self.assertAlmostEqual(result["soc"], 0.6125)

    def test_parse_warmup_result(self):
        result = serial_benchmark.parse_result("RESULT,17,DD,nan,0,WARMUP", "17", "DD")
        self.assertIsNone(result["soc"])
        self.assertEqual(result["status"], "WARMUP")

    def test_vector_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.csv"
            fields = ["sample_id", "segment_id", "reset", *serial_benchmark.INPUT_COLUMNS, "expected_dd"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({name: "1" for name in fields})
            rows = serial_benchmark.load_vectors(path)
            self.assertEqual(len(rows), 1)

    def test_result_schema_contains_dataset_comparison(self):
        self.assertIn("soc_dataset", serial_benchmark.RESULT_COLUMNS)
        self.assertIn("dataset_error", serial_benchmark.RESULT_COLUMNS)
        self.assertIn("dataset_abs_error", serial_benchmark.RESULT_COLUMNS)


class MemoryReportTests(unittest.TestCase):
    def test_parse_sections(self):
        sections = memory_report.parse_size_output(
            "image.elf  :\nsection size addr\n.text 1024 0x08000000\n.bss 256 0x20000000\n"
        )
        self.assertEqual(sections, {".text": 1024, ".bss": 256})


if __name__ == "__main__":
    unittest.main()
