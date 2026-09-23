"""Tests for the CFR iteration stamped on Deep CFR reservoir samples (cambia-720).

reservoir.py documents ``ReservoirSample.iteration`` as "CFR iteration number
for weighting", and the fit loop weights each sample by (iteration + 1)^alpha
normalised by the batch mean. Every traversal-dispatch path in deep_trainer.py
stamped a running traversal counter instead, so samples drawn from one policy
inside a single training step received weights spanning a ratio of up to
traversals_per_step^alpha. These tests pin the value each dispatch path hands
the worker: one training step, one iteration number.

The worker itself is stubbed. deep_worker.py stores the iteration it is handed
and is owned by another lane, so the value has to be right at the dispatch
sites, which is what these tests measure.
"""

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.reservoir import ReservoirSample
from src.cfr.deep_trainer import DeepCFRConfig, DeepCFRTrainer

ALPHA = 1.5


def _make_config(num_workers=1):
    config = SimpleNamespace()
    config.cfr_training = SimpleNamespace(num_iterations=2, num_workers=num_workers)
    config.persistence = SimpleNamespace(agent_data_save_path="test_ckpt.pt")
    config.logging = SimpleNamespace(
        log_level_file="WARNING",
        log_level_console="WARNING",
        log_dir="/tmp/test_logs",
        log_file_prefix="cambia_test",
        log_max_bytes=1024 * 1024,
        log_backup_count=1,
        log_simulation_traces=False,
        log_archive_enabled=False,
        get_worker_log_level=lambda wid, ntotal: "WARNING",
    )
    return config


def _dcfr_config(**overrides):
    base = dict(
        device="cpu",
        hidden_dim=8,
        num_hidden_layers=1,
        num_traversal_threads=1,
        engine_backend="go",
        traversals_per_step=3,
        train_steps_per_iteration=0,
        batch_size=4,
        save_interval=0,
        es_validation_interval=0,
        pipeline_training=False,
        advantage_buffer_capacity=1000,
        strategy_buffer_capacity=1000,
        alpha=ALPHA,
    )
    base.update(overrides)
    return DeepCFRConfig(**base)


def _sample(iteration):
    return ReservoirSample(
        features=np.zeros(INPUT_DIM, dtype=np.float32),
        target=np.zeros(NUM_ACTIONS, dtype=np.float32),
        action_mask=np.ones(NUM_ACTIONS, dtype=bool),
        iteration=iteration,
    )


class _Stats:
    nodes_visited = 1
    error_count = 0
    max_depth = 1


def _stub_worker(args, file_handler_override=None):
    """Return one advantage and one strategy sample carrying the handed iteration."""
    from src.cfr.deep_trainer import DeepCFRWorkerResult

    iteration = args[0]
    return DeepCFRWorkerResult(
        advantage_samples=[_sample(iteration)],
        strategy_samples=[_sample(iteration)],
        stats=_Stats(),
    )


def _iterations_by_step(trainer):
    """Advantage-buffer iteration values in insertion order."""
    return [
        int(v)
        for v in trainer.advantage_buffer._iterations[: len(trainer.advantage_buffer)]
    ]


def _run(monkeypatch, dcfr_config, num_workers=1, steps=2):
    import src.cfr.deep_trainer as dt

    monkeypatch.setattr(dt, "run_deep_cfr_worker", _stub_worker)
    trainer = DeepCFRTrainer(
        config=_make_config(num_workers=num_workers), deep_cfr_config=dcfr_config
    )
    trainer.train(num_training_steps=steps)
    return trainer


def _assert_one_iteration_per_step(iterations, traversals_per_step, steps):
    """Each step contributes traversals_per_step samples sharing one iteration."""
    assert len(iterations) == traversals_per_step * steps
    per_step = [
        iterations[i * traversals_per_step : (i + 1) * traversals_per_step]
        for i in range(steps)
    ]
    for block in per_step:
        assert len(set(block)) == 1, f"one step spans iterations {sorted(set(block))}"
    return [block[0] for block in per_step]


def _weight_ratio(t_hi, t_lo):
    return (t_hi + 1.0) ** ALPHA / (t_lo + 1.0) ** ALPHA


class TestSequentialDispatch:
    def test_two_steps_stamp_one_iteration_each(self, monkeypatch):
        trainer = _run(monkeypatch, _dcfr_config())

        t1, t2 = _assert_one_iteration_per_step(_iterations_by_step(trainer), 3, 2)

        assert (t1, t2) == (1, 2)
        assert _weight_ratio(t2, t1) == pytest.approx((3.0 / 2.0) ** ALPHA)


class TestThreadedDispatch:
    def test_two_steps_stamp_one_iteration_each(self, monkeypatch):
        trainer = _run(monkeypatch, _dcfr_config(num_traversal_threads=2))

        t1, t2 = _assert_one_iteration_per_step(_iterations_by_step(trainer), 3, 2)

        assert (t1, t2) == (1, 2)


class TestPooledDispatch:
    def test_two_steps_stamp_one_iteration_each(self, monkeypatch):
        """The multi-worker path stamps the step, not the pool slot index."""
        import multiprocessing

        import src.cfr.deep_trainer as dt

        created = []

        class _InlinePool:
            def __init__(self, processes=None):
                self.processes = processes
                created.append(processes)

            def map_async(self, fn, args_list):
                results = [_stub_worker(a) for a in args_list]

                class _Async:
                    def ready(self_inner):
                        return True

                    def wait(self_inner, timeout=None):
                        return None

                    def get(self_inner):
                        return results

                return _Async()

            def close(self):
                pass

            def terminate(self):
                pass

            def join(self):
                pass

        # The trainer takes its pool from the spawn context (cambia-2402), so the
        # stand-in goes on that context; a pool from anywhere else never reaches
        # it, which the `created` check below catches.
        monkeypatch.setattr(
            type(multiprocessing.get_context("spawn")), "Pool", _InlinePool
        )
        trainer = _run(monkeypatch, _dcfr_config(), num_workers=2)

        assert created == [2], "the pooled path did not build its pool from spawn"
        t1, t2 = _assert_one_iteration_per_step(_iterations_by_step(trainer), 3, 2)

        assert (t1, t2) == (1, 2)


class TestPipelinedDispatch:
    def test_prefetched_traversals_carry_the_step_they_feed(self, monkeypatch):
        """Work submitted during step N is consumed at step N+1, so it is N+1."""
        import src.cfr.deep_trainer as dt

        class _InlineFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

            def cancel(self):
                return True

        class _InlineExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def submit(self, fn, *args, **kwargs):
                return _InlineFuture(fn(*args, **kwargs))

            def shutdown(self, wait=True):
                pass

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
        trainer = _run(monkeypatch, _dcfr_config(pipeline_training=True), steps=3)

        iterations = _iterations_by_step(trainer)
        steps_seen = _assert_one_iteration_per_step(iterations, 3, 3)

        assert steps_seen == [1, 2, 3]


class TestBatchHelperStampsOneIteration:
    def test_every_traversal_in_a_batch_shares_the_iteration(self, monkeypatch):
        """_run_traversals_batch stamps its iteration argument, not offset + i."""
        import src.cfr.deep_trainer as dt

        monkeypatch.setattr(dt, "run_deep_cfr_worker", _stub_worker)

        adv, strat, _values, done, _nodes, _timing = dt._run_traversals_batch(
            7,
            _make_config(),
            {},
            {},
            4,
            1,
            "logs",
            "test",
        )

        assert done == 4
        assert [s.iteration for s in adv] == [7, 7, 7, 7]
        assert [s.iteration for s in strat] == [7, 7, 7, 7]


class TestResumeMigratesPreFixIterations:
    """A checkpoint written before this fix holds per-traversal iterations.

    Left alone they mix with step numbers in one buffer, and since the weight
    is (t + 1)^1.5 a stored 100000 outweighs a fresh step 101 by roughly 3e4,
    so the resumed run would train almost entirely on its old samples. That is
    far worse than the intra-step spread this ticket removes, so the values are
    converted to the step that produced them on load.
    """

    def _trainer(self, tmp_path, **overrides):
        cfg = _dcfr_config(**overrides)
        trainer = DeepCFRTrainer(config=_make_config(), deep_cfr_config=cfg)
        return trainer

    def test_per_traversal_values_become_step_numbers(self, tmp_path):
        trainer = self._trainer(tmp_path, traversals_per_step=100)
        trainer.training_step = 3
        trainer.total_traversals = 300
        # Pre-fix stamping: a running traversal counter across three steps.
        for traversal_index in (0, 99, 100, 199, 200, 299):
            trainer.advantage_buffer.add(_sample(traversal_index))

        trainer._migrate_pre_step_iterations(traversals_per_step=100)

        migrated = _iterations_by_step(trainer)
        assert migrated == [1, 1, 2, 2, 3, 3]

    def test_already_migrated_buffer_is_left_alone(self, tmp_path):
        trainer = self._trainer(tmp_path, traversals_per_step=100)
        trainer.training_step = 3
        trainer.total_traversals = 300
        for step in (1, 1, 2, 3):
            trainer.advantage_buffer.add(_sample(step))

        trainer._migrate_pre_step_iterations(traversals_per_step=100)

        assert _iterations_by_step(trainer) == [1, 1, 2, 3]

    def test_checkpoint_round_trip_migrates(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        writer = self._trainer(tmp_path, traversals_per_step=100)
        writer.training_step = 2
        writer.total_traversals = 200
        for traversal_index in (0, 50, 150):
            writer.advantage_buffer.add(_sample(traversal_index))
        writer.save_checkpoint(path)

        reader = self._trainer(tmp_path, traversals_per_step=100)
        reader.load_checkpoint(path)

        assert max(_iterations_by_step(reader)) <= reader.training_step
        assert sorted(_iterations_by_step(reader)) == [1, 1, 2]
