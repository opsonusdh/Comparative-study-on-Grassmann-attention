import torch

from src.loop_transformer import LoopConfig, LoopTransformer


def tiny_cfg(dropout=0.0):
    return LoopConfig(
        vocab_size=32, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_hidden_dim=64, rope_dim=8, max_loops=2, min_loops=2,
        csa_m=2, csa_top_k=3, hca_m_prime=3, sw_window=6,
        groups=2, group_dim=16, moe_num_shared_experts=1,
        moe_num_routed_experts=2, moe_top_k=1, activation_top_k=2,
        loop_sampling=False,
        dropout=dropout, tie_embeddings=True, grad_checkpointing=False,
    )


def test_dropout_is_configured_and_disabled_is_legacy_safe():
    model = LoopTransformer(tiny_cfg(0.0))
    assert model.cfg.dropout == 0.0
    assert all(block.dropout.p == 0.0 for block in model.blocks)


def test_dropout_training_changes_sub_layer_branch_but_eval_disables_it():
    model = LoopTransformer(tiny_cfg(0.5))
    assert all(block.dropout.p == 0.5 for block in model.blocks)
    x = torch.ones(2, 8, 32)
    model.train()
    torch.manual_seed(123)
    a = model.blocks[0].dropout(x)
    torch.manual_seed(123)
    b = model.blocks[0].dropout(x)
    assert torch.equal(a, b)
    assert (a == 0).any()
    model.eval()
    y = model.blocks[0].dropout(x)
    assert torch.equal(y, x)


def test_dropout_forward_backward_is_finite():
    torch.manual_seed(0)
    model = LoopTransformer(tiny_cfg(0.1))
    model.train()
    idx = torch.randint(0, 32, (2, 12))
    loss, _ = model.compute_loss(idx, max_loops=2)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
