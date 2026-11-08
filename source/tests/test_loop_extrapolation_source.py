from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = (ROOT / "src" / "loop_transformer" / "model.py").read_text()
GENERATE = (ROOT / "generate.py").read_text()


def test_forward_has_explicit_extrapolation_switch():
    assert "allow_loop_extrapolation: bool = False" in MODEL
    assert "if max_loops > self.cfg.max_loops:" in MODEL
    assert "allow_loop_extrapolation" in MODEL


def test_terminal_hold_prevents_out_of_range_loop_embedding():
    assert "conditioning_loop_idx = min(int(loop_idx), self.cfg.max_loops - 1)" in MODEL


def test_generation_requires_explicit_flag():
    assert "--extrapolate-loops" in GENERATE
    assert "allow_loop_extrapolation=args.extrapolate_loops" in GENERATE


def test_transition_matrices_are_not_falsely_extrapolated():
    assert "len(self.loop_transition_matrices) > 0" in MODEL
    assert "not supported when explicit" in MODEL
