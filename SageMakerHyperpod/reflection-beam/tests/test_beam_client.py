from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import beam_client


class BaseUrlTests(unittest.TestCase):
    def test_accepts_http_and_https_urls(self) -> None:
        self.assertEqual(
            beam_client.validate_base_url("http://127.0.0.1:8000/v1/"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            beam_client.validate_base_url("https://beam.example/v1"),
            "https://beam.example/v1",
        )

    def test_rejects_invalid_or_credentialed_urls(self) -> None:
        for value in ("localhost:8000/v1", "ftp://example.test", "https://u:p@x/v1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    beam_client.validate_base_url(value)

    def test_missing_dependency_has_actionable_error(self) -> None:
        with mock.patch.object(beam_client, "OpenAI", None):
            with self.assertRaisesRegex(RuntimeError, "requirements.txt"):
                beam_client.build_client("http://127.0.0.1:8000/v1", 30)

    def test_authenticated_remote_endpoint_requires_https(self) -> None:
        with (
            mock.patch.object(beam_client, "OpenAI", mock.Mock()),
            mock.patch.dict(
                beam_client.os.environ,
                {"BEAM_API_KEY": "short-lived-token"},
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                beam_client.build_client("http://beam.example/v1", 30)

            beam_client.build_client("http://127.0.0.1:8000/v1", 30)
            beam_client.build_client("https://beam.example/v1", 30)


class WorkspaceBoundaryTests(unittest.TestCase):
    def test_reads_only_supported_files_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "example.py"
            target.write_text("print('ok')\n", encoding="utf-8")
            self.assertEqual(
                beam_client.resolve_workspace_file(root, "example.py"),
                target.resolve(),
            )

    def test_rejects_path_escape_and_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            root.mkdir()
            outside = Path(temp_dir) / "outside.py"
            outside.write_text("pass\n", encoding="utf-8")
            binary = root / "data.bin"
            binary.write_bytes(b"\x00")

            with self.assertRaisesRegex(ValueError, "escapes"):
                beam_client.resolve_workspace_file(root, "../outside.py")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                beam_client.resolve_workspace_file(root, "data.bin")

    def test_rejects_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "large.py"
            target.write_bytes(b"x" * (beam_client.MAX_FILE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                beam_client.resolve_workspace_file(root, "large.py")


class ArgumentTests(unittest.TestCase):
    def test_positive_bounds(self) -> None:
        args = beam_client.parse_args(
            [
                "chat",
                "--max-tokens",
                "1",
                "--timeout-seconds",
                "1",
                "--temperature",
                "2",
                "--top-p",
                "0.7",
            ]
        )
        self.assertEqual(args.max_tokens, 1)
        self.assertEqual(args.timeout_seconds, 1)
        self.assertEqual(args.top_p, 0.7)

    def test_non_positive_max_tokens_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                beam_client.parse_args(["chat", "--max-tokens", "0"])

    def test_invalid_top_p_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                beam_client.parse_args(["chat", "--top-p", "1.1"])


if __name__ == "__main__":
    unittest.main()
