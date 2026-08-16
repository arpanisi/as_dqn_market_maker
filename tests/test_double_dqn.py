import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agent.double_dqn import DoubleDQNAgent, QNetwork, Transition, action_space, build_state


def test_action_space_is_joint_gamma_skew_grid():
    actions = action_space()

    assert len(actions) == 20
    assert len({action.risk_aversion for action in actions}) == 4
    assert len({action.skew for action in actions}) == 5
    assert actions[0].risk_aversion == 0.01
    assert actions[-1].skew == 0.1


def test_state_includes_funding_normalized_by_one_percent():
    state = build_state(
        inventory_q=2.0,
        realized_volatility=0.5,
        arrival_A=10.0,
        arrival_kappa=0.2,
        time_remaining_fraction=1.2,
        adverse_selection=0.3,
        funding_rate=0.0002,
    )

    assert state.shape == (7,)
    assert state[4] == 1.0
    assert state[6] == 0.02


def test_double_dqn_learn_step_after_small_test_warmup():
    agent = DoubleDQNAgent(seed=7, warmup_transitions=2, batch_size=2)
    state = np.zeros(7)
    next_state = np.ones(7) * 0.1

    assert agent.learn_from_transition(Transition(state, 0, 1.0, next_state, False)) is None
    loss = agent.learn_from_transition(Transition(next_state, 1, 0.5, state, True))

    assert loss is not None
    assert loss >= 0.0


def test_q_network_matches_locked_mlp_shape():
    network = QNetwork(state_dim=7, action_dim=20, hidden_units=128)

    layers = list(network.net)
    assert layers[0].in_features == 7
    assert layers[0].out_features == 128
    assert layers[2].out_features == 128
    assert layers[4].out_features == 20


def test_agent_save_and_load_roundtrip(tmp_path):
    agent = DoubleDQNAgent(seed=7, warmup_transitions=2, batch_size=2, device="cpu")
    path = tmp_path / "weights.pt"

    agent.save(path)
    loaded = DoubleDQNAgent(seed=8, warmup_transitions=2, batch_size=2, device="cpu")
    loaded.load(path)

    state = np.zeros(7, dtype=np.float32)
    assert loaded.select_action(state).index == agent.select_action(state).index
