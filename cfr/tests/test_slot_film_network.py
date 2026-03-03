"""
tests/test_slot_film_network.py

Unit tests for SlotFiLMAdvantageNetwork architecture.

Layout: [public(42)][12 × slot(13)][pad(2)][history(24)] = 224 dims
  slot(13) = tag(4) + identity(9)
  slot offset = 42 + slot_index * 13
"""

import pytest
import torch
import torch.nn as nn
from src.constants import EP_PBS_INPUT_DIM
from src.networks import SlotFiLMAdvantageNetwork, build_advantage_network


class TestSlotFiLMNetwork:
    """Tests for SlotFiLMAdvantageNetwork architecture."""

    def test_forward_shape(self):
        """Output shape is (batch, num_actions) with correct masking."""
        net = SlotFiLMAdvantageNetwork()
        x = torch.randn(8, EP_PBS_INPUT_DIM)
        mask = torch.ones(8, 146, dtype=torch.bool)
        mask[:, 100:] = False  # mask out some actions
        out = net(x, mask)
        assert out.shape == (8, 146)
        assert (out[:, 100:] == float("-inf")).all()

    def test_param_count(self):
        """Param count should be ~490K-520K."""
        net = SlotFiLMAdvantageNetwork()
        params = sum(p.numel() for p in net.parameters())
        assert 490_000 < params < 520_000, f"Expected ~505K params, got {params}"

    def test_film_init_zeros(self):
        """FiLM gamma and beta weights and biases must be zero at init."""
        net = SlotFiLMAdvantageNetwork()
        assert net.film_gamma.weight.abs().max().item() == 0.0, (
            "film_gamma.weight not zero-initialized"
        )
        assert net.film_gamma.bias.abs().max().item() == 0.0, (
            "film_gamma.bias not zero-initialized"
        )
        assert net.film_beta.weight.abs().max().item() == 0.0, (
            "film_beta.weight not zero-initialized"
        )
        assert net.film_beta.bias.abs().max().item() == 0.0, (
            "film_beta.bias not zero-initialized"
        )

    def test_tag_id_bias_zeros(self):
        """tag_embed and id_embed biases must be zero at init (preserves empty/UNK divergence)."""
        net = SlotFiLMAdvantageNetwork()
        assert net.tag_embed.bias.abs().max().item() == 0.0, (
            "tag_embed.bias not zero-initialized"
        )
        assert net.id_embed.bias.abs().max().item() == 0.0, (
            "id_embed.bias not zero-initialized"
        )

    def test_identity_passthrough_at_init(self):
        """At init, gated == id_h since gamma_delta=0 and beta=0."""
        net = SlotFiLMAdvantageNetwork()
        net.eval()
        with torch.no_grad():
            # Single slot
            tag = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # UNK tag (one-hot index 0)
            identity = torch.randn(1, 9)
            tag_h = torch.relu(net.tag_embed(tag))
            id_h = torch.relu(net.id_embed(identity))
            gamma_delta = net.film_gamma(tag_h)
            beta = net.film_beta(tag_h)
            gated = (1.0 + gamma_delta) * id_h + beta
        # gamma_delta=0 and beta=0 at init → gated should equal id_h
        assert torch.allclose(gated, id_h, atol=1e-6), (
            f"Expected gated == id_h at init; max diff = {(gated - id_h).abs().max().item()}"
        )

    def test_priv_opp_gradient_flow(self):
        """Gradient flows through beta (additive) path when identity is all-zeros (PRIV_OPP case).

        This is the critical test: when identity=0, the multiplicative path (1+gamma)*id_h
        produces a zero output AND zero gradient back through id_h. FiLM's additive beta term
        is the ONLY gradient path from the loss to the tag encoder. If beta has no gradient,
        the network can't learn anything from PRIV_OPP slots.
        """
        net = SlotFiLMAdvantageNetwork()
        # Build input: PRIV_OPP tag in slot 0, identity all-zeros
        # slot 0 starts at offset 42; tag occupies [42:46], identity [46:55]
        x = torch.zeros(2, EP_PBS_INPUT_DIM, requires_grad=False)
        # PRIV_OPP = tag index 3 → one-hot [0, 0, 0, 1]
        x[:, 42 + 3] = 1.0  # slot 0 tag dim 3
        # identity dims 46..54 stay zero

        mask = torch.ones(2, 146, dtype=torch.bool)
        out = net(x, mask)
        loss = out.sum()
        loss.backward()

        # film_beta must receive gradient (it is the only path when id_h=0)
        assert net.film_beta.weight.grad is not None, "film_beta.weight has no gradient"
        assert net.film_beta.weight.grad.abs().max().item() > 0.0, (
            "film_beta.weight gradient is zero — additive path is broken"
        )

    def test_empty_slot_film_beta_weight_grad_zero(self):
        """Empty slots (all-zeros input including tag) produce zero weight gradient on film_beta.

        With all-zeros tag input and zero bias, tag_h=relu(W*0+0)=0.
        film_beta(tag_h) = W_beta * 0 + b_beta = b_beta (constant).
        The gradient of the loss w.r.t. W_beta is tag_h (=0), so W_beta.grad == 0.
        This confirms empty and PRIV_OPP slots diverge after the first SGD step
        (bias update only for empty; weight+bias update for PRIV_OPP).
        """
        net = SlotFiLMAdvantageNetwork()
        # All-zeros input → all slots are "empty"
        x = torch.zeros(2, EP_PBS_INPUT_DIM)
        mask = torch.ones(2, 146, dtype=torch.bool)
        out = net(x, mask)
        loss = out.sum()
        loss.backward()
        assert net.film_beta.weight.grad is not None, "film_beta.weight has no gradient"
        assert net.film_beta.weight.grad.abs().max().item() == 0.0, (
            "film_beta.weight should have zero gradient for all-zero (empty) input"
        )

    def test_slot_multiply_no_priv_opp_gradient(self):
        """Ablation B (multiply gate): zero gradient on tag_gate.weight when identity is all-zeros.

        Confirms FiLM's structural necessity: without the additive beta path,
        PRIV_OPP slots have no gradient at all.
        """
        net = SlotFiLMAdvantageNetwork(use_film=False)
        x = torch.zeros(2, EP_PBS_INPUT_DIM)
        x[:, 42 + 3] = 1.0  # PRIV_OPP tag at slot 0
        # identity stays zero → id_h = 0 → gate * id_h = 0 → no gradient through gate
        mask = torch.ones(2, 146, dtype=torch.bool)
        out = net(x, mask)
        loss = out.sum()
        loss.backward()
        assert net.tag_gate.weight.grad is not None, "tag_gate.weight has no gradient"
        assert net.tag_gate.weight.grad.abs().max().item() == 0.0, (
            "tag_gate.weight should have zero gradient when id_h=0 (multiply gate dead-end)"
        )

    def test_no_pos_embed_variant(self):
        """use_pos_embed=False variant produces correct output shape."""
        net = SlotFiLMAdvantageNetwork(use_pos_embed=False)
        x = torch.randn(4, EP_PBS_INPUT_DIM)
        mask = torch.ones(4, 146, dtype=torch.bool)
        out = net(x, mask)
        assert out.shape == (4, 146)

    def test_factory_dispatch_slot_film(self):
        """Factory returns SlotFiLMAdvantageNetwork for network_type='slot_film'."""
        net = build_advantage_network(
            input_dim=EP_PBS_INPUT_DIM,
            network_type="slot_film",
            num_hidden_layers=3,
        )
        assert isinstance(net, SlotFiLMAdvantageNetwork), (
            f"Expected SlotFiLMAdvantageNetwork, got {type(net).__name__}"
        )

    def test_factory_dispatch_slot_multiply(self):
        """Factory returns SlotFiLMAdvantageNetwork(use_film=False) for 'slot_multiply'."""
        net = build_advantage_network(input_dim=EP_PBS_INPUT_DIM, network_type="slot_multiply")
        assert isinstance(net, SlotFiLMAdvantageNetwork), (
            f"Expected SlotFiLMAdvantageNetwork, got {type(net).__name__}"
        )
        assert not net.use_film, "use_film should be False for slot_multiply"

    def test_factory_backward_compat(self):
        """Factory still returns ResidualAdvantageNetwork for use_residual=True."""
        from src.networks import ResidualAdvantageNetwork

        net = build_advantage_network(use_residual=True)
        assert isinstance(net, ResidualAdvantageNetwork), (
            f"Expected ResidualAdvantageNetwork, got {type(net).__name__}"
        )

    def test_illegal_actions_are_neg_inf(self):
        """All illegal actions in the mask produce exactly -inf in output."""
        net = SlotFiLMAdvantageNetwork()
        x = torch.randn(4, EP_PBS_INPUT_DIM)
        mask = torch.zeros(4, 146, dtype=torch.bool)
        mask[:, :10] = True  # only first 10 legal
        out = net(x, mask)
        assert (out[:, 10:] == float("-inf")).all(), (
            "Expected -inf for all illegal actions"
        )
        # Legal actions should not be -inf
        assert not (out[:, :10] == float("-inf")).any(), (
            "Legal actions should not be -inf"
        )

    def test_deterministic_eval_mode(self):
        """Two identical forward passes in eval mode produce identical outputs (dropout off)."""
        net = SlotFiLMAdvantageNetwork()
        net.eval()
        x = torch.randn(4, EP_PBS_INPUT_DIM)
        mask = torch.ones(4, 146, dtype=torch.bool)
        with torch.no_grad():
            out1 = net(x, mask)
            out2 = net(x, mask)
        assert torch.equal(out1, out2), "Eval mode must be deterministic"
