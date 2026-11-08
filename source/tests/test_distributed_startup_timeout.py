from datetime import timedelta
from types import SimpleNamespace

import train


def test_set_distributed_timeout_uses_public_api(monkeypatch):
    calls = []

    class FakeDist:
        group = SimpleNamespace(WORLD="world-group")

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def set_timeout(timeout, group=None):
            calls.append((timeout, group))

    monkeypatch.setattr(train, "dist", FakeDist)

    assert train._set_distributed_timeout(
        1800,
        reason="unit test",
    ) is True
    assert calls == [(timedelta(seconds=1800), "world-group")]


def test_set_distributed_timeout_uses_legacy_fallback(monkeypatch):
    calls = []

    class FakeDist:
        group = SimpleNamespace(WORLD="world-group")

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

    def legacy_setter(timeout, group=None):
        calls.append((timeout, group))

    import torch.distributed.distributed_c10d as c10d
    monkeypatch.setattr(train, "dist", FakeDist)
    monkeypatch.setattr(c10d, "_set_pg_timeout", legacy_setter)

    assert train._set_distributed_timeout(
        120,
        reason="unit test",
    ) is True
    assert calls == [(timedelta(seconds=120), "world-group")]


def test_set_distributed_timeout_is_noop_without_initialized_group(monkeypatch):
    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return False

    monkeypatch.setattr(train, "dist", FakeDist)

    assert train._set_distributed_timeout(
        120,
        reason="unit test",
    ) is False


def test_kaggle_restore_timeout_default_and_validation(monkeypatch):
    # Keep the environment variable's documented default explicit in the code
    # path by testing the same parsing contract used by _run_from_cli.
    monkeypatch.delenv("LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS", raising=False)
    seconds = int(
        __import__("os").environ.get(
            "LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS", "1800"
        )
    )
    assert seconds == 1800

    monkeypatch.setenv("LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS", "2400")
    assert int(__import__("os").environ["LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS"]) == 2400
