from argparse import Namespace
from pathlib import Path

import pytest
import torch

from train import _validate_resume_checkpoint, _validate_resume_schedule_overrides, scheduled_lr


def _valid_checkpoint(step: int = 600):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2.95e-5, betas=(0.9, 0.95))
    parameter.square().sum().backward()
    optimizer.step()
    return {
        "step": step,
        "model_state_dict": {"weight": torch.ones(2)},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {},
        "extra": {
            "lr_at_checkpoint": 2.95e-5,
            "lr_state": {
                "mode": "normal",
                "base_lr": 3e-5,
                "min_lr_ratio": 0.05,
                "max_steps": 5000,
            },
            "args": {
                "lr": 3e-5,
                "warmup_steps": 200,
                "min_lr_ratio": 0.05,
                "max_steps": 5000,
            },
        },
    }


def test_resume_requires_positive_completed_step(tmp_path: Path):
    checkpoint = _valid_checkpoint(step=0)
    with pytest.raises(RuntimeError, match="invalid completed step"):
        _validate_resume_checkpoint(tmp_path / "latest.pt", checkpoint)


def test_resume_requires_optimizer_state(tmp_path: Path):
    checkpoint = _valid_checkpoint()
    checkpoint.pop("optimizer_state_dict")
    with pytest.raises(RuntimeError, match="optimizer_state_dict is missing"):
        _validate_resume_checkpoint(tmp_path / "latest.pt", checkpoint)


def test_resume_requires_saved_lr_schedule_metadata(tmp_path: Path):
    checkpoint = _valid_checkpoint()
    checkpoint["extra"].pop("lr_state")
    checkpoint["extra"].pop("lr_schedule", None)
    with pytest.raises(RuntimeError, match="no recognized LR schedule state"):
        _validate_resume_checkpoint(tmp_path / "latest.pt", checkpoint)


def test_valid_resume_reports_saved_step(tmp_path: Path):
    assert _validate_resume_checkpoint(tmp_path / "latest.pt", _valid_checkpoint(600)) == 600


def test_ordinary_resume_rejects_accidental_lr_restart():
    checkpoint = _valid_checkpoint()
    args = Namespace(
        lr=2e-3,
        warmup_steps=100,
        min_lr_ratio=0.05,
        max_steps=5000,
        force_lr=None,
        _explicit_cli_overrides={"lr", "warmup_steps"},
    )
    with pytest.raises(ValueError, match="Cannot change --lr"):
        _validate_resume_schedule_overrides(args, checkpoint)


def test_force_lr_allows_explicit_new_schedule():
    checkpoint = _valid_checkpoint()
    args = Namespace(
        lr=3e-5,
        warmup_steps=0,
        min_lr_ratio=0.05,
        max_steps=10000,
        force_lr=3e-5,
        _explicit_cli_overrides={"lr", "warmup_steps", "max_steps"},
    )
    _validate_resume_schedule_overrides(args, checkpoint)


def test_warmup_cosine_schedule_is_anchored_to_global_step():
    args = Namespace(lr=3e-5, warmup_steps=200, max_steps=5000, min_lr_ratio=0.05)
    assert scheduled_lr(200, args) == pytest.approx(3e-5)
    assert scheduled_lr(5000, args) == pytest.approx(1.5e-6)
    # A resume at 600 must use the LR for global step 601, not warm up again.
    assert scheduled_lr(601, args) < scheduled_lr(600, args)
    assert scheduled_lr(601, args) > scheduled_lr(5000, args)


def test_require_resume_flag_rejects_missing_checkpoint():
    from train import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--require-resume"])


def test_require_resume_flag_accepts_explicit_checkpoint_path():
    from train import parse_args

    args = parse_args(["--resume", "/tmp/latest.pt", "--require-resume"])
    assert args.resume == Path("/tmp/latest.pt")
    assert args.require_resume is True
