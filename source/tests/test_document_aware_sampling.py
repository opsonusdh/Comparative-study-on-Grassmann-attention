from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("loopllm_train", ROOT / "train.py")
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


class DocumentAwareSamplingTest(unittest.TestCase):
    def test_document_index_never_samples_an_eot(self) -> None:
        eot = 99
        # Eligible spans are [0, 4) and [5, 10); the final span is too short.
        data = np.asarray([1, 2, 3, 4, eot, 5, 6, 7, 8, 9, eot, 10, 11], dtype=np.uint16)
        index = train.build_document_window_index(data, seq_len=2, eot_token_id=eot)
        self.assertEqual(index.total_windows, 5)
        starts = index.sample_starts(200, np.random.default_rng(7))
        self.assertTrue(all(eot not in data[start:start + 3] for start in starts))

    def test_short_documents_are_skipped_without_duplication(self) -> None:
        data = np.asarray([1, 2, 99, 3, 4, 5, 6, 99], dtype=np.uint16)
        index = train.build_document_window_index(data, seq_len=3, eot_token_id=99)
        self.assertEqual(index.total_windows, 1)
        self.assertEqual(index.sample_starts(1, np.random.default_rng(1)).tolist(), [3])

    def test_eot_mask_excludes_only_the_target_after_eot(self) -> None:
        batch = torch.tensor([[1, 99, 2, 3], [99, 4, 5, 6]])
        self.assertEqual(train.eot_cross_document_target_mask(batch, 99).tolist(), [
            [True, False, True], [False, True, True]
        ])

    def test_target_mask_is_supported_by_the_model_loss(self) -> None:
        from loop_transformer import LoopConfig, LoopTransformer

        cfg = LoopConfig(
            vocab_size=64, dim=32, n_layers=1, n_heads=2, head_dim=16,
            ffn_hidden_dim=64, rope_dim=16, max_loops=2, min_loops=2,
            sw_window=4, csa_m=2, csa_top_k=1, hca_m_prime=4, groups=1,
            group_dim=32, moe_num_shared_experts=1, moe_num_routed_experts=1,
            moe_top_k=1, moe_expert_hidden_dim=32, dropout=0.0,
            causal_register_bank=False,
        )
        model = LoopTransformer(cfg).eval()
        batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mask = torch.tensor([[True, False, True], [True, True, False]])
        loss, _ = model.compute_loss(batch, max_loops=2, target_mask=mask)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "excludes every next-token target"):
            model.compute_loss(batch, max_loops=2, target_mask=torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
