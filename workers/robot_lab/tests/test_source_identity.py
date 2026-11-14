"""The service tells a paired application which code it runs."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source_identity import source_identity  # noqa: E402


class SourceIdentityTests(unittest.TestCase):
    def test_the_fingerprint_follows_code_and_ignores_notes_tests_and_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder)
            (tree / "server.py").write_text("print('one')\n")
            (tree / "requirements-hardware.txt").write_text("pyserial==3.5\n")
            first = source_identity(tree)
            (tree / "README.md").write_text("notes")
            (tree / "tests").mkdir()
            (tree / "tests/test_server.py").write_text("assert True\n")
            (tree / "__pycache__").mkdir()
            (tree / "__pycache__/server.cpython-314.pyc").write_bytes(b"cache")
            self.assertEqual(source_identity(tree), first)
            (tree / "server.py").write_text("print('two')\n")
            self.assertNotEqual(source_identity(tree), first)

    def test_the_running_service_reports_the_fingerprint_of_its_own_folder(self):
        import server

        self.assertEqual(server.SOURCE_SHA256, source_identity())
        self.assertRegex(server.SOURCE_SHA256, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
