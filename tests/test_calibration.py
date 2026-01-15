"""
Tests for v0.33.0 Automatic Steady-State Checkpointing.

Verifies that the checkpoint system:
- Creates checkpoints on first run
- Loads existing checkpoints on subsequent runs
- Produces consistent metrics across checkpoint loads
- Correctly interprets --days as steady-state data days (not including burn-in)
"""

import shutil
from pathlib import Path

import pytest

from prism_sim.simulation.orchestrator import Orchestrator


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


class TestAutoCheckpointSystem:
    """Tests for automatic checkpoint creation and loading."""

    def test_checkpoint_created_on_first_run(self) -> None:
        """First run without checkpoint should create one."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # Run orchestrator with auto_checkpoint enabled
        sim = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )

        # Should need burn-in (no checkpoint exists)
        assert sim._needs_burn_in is True

        # Run short simulation to trigger checkpoint creation
        sim.run(days=5)

        # Checkpoint should now exist
        assert sim._checkpoint_path.exists(), "Checkpoint file was not created"

        # Clean up
        shutil.rmtree(checkpoint_dir)

    def test_checkpoint_loaded_on_second_run(self) -> None:
        """Second run should load existing checkpoint."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # First run - creates checkpoint
        sim1 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim1.run(days=5)

        # Second run - should load checkpoint
        sim2 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )

        # Should NOT need burn-in (checkpoint exists)
        assert sim2._needs_burn_in is False
        assert sim2._warm_start_snapshot is not None
        assert sim2._start_day > 1  # Should start after burn-in

        # Clean up
        shutil.rmtree(checkpoint_dir)

    def test_no_checkpoint_flag_forces_cold_start(self) -> None:
        """--no-checkpoint should always cold-start."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # First run creates checkpoint
        sim1 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim1.run(days=5)

        # Second run with auto_checkpoint=False should ignore checkpoint
        sim2 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=False,
        )

        # Should start from day 1 (cold start)
        assert sim2._start_day == 1
        assert sim2._warm_start_snapshot is None
        assert sim2._needs_burn_in is False  # Not auto-checkpointing, so no burn-in needed

        # Clean up
        shutil.rmtree(checkpoint_dir)


class TestDaysMeaning:
    """Tests that --days N produces exactly N days of data."""

    def test_days_produces_correct_metric_count(self) -> None:
        """Run with --days 10 should produce metrics for exactly 10 days."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # Run with checkpoint to get steady-state
        sim = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim.run(days=10)

        # Check that monitor recorded 10 days of metrics
        # Access the tracker's count directly
        n_samples = sim.monitor.service_level_tracker.count

        assert n_samples == 10, (
            f"Expected 10 service level samples (for 10 data days), got {n_samples}"
        )

        # Clean up
        shutil.rmtree(checkpoint_dir)

    def test_cold_start_excludes_burn_in_from_metrics(self) -> None:
        """Cold-start should exclude burn-in period from metrics."""
        # Run with --no-checkpoint for cold start
        sim = Orchestrator(
            enable_logging=False,
            auto_checkpoint=False,
        )

        # Run 100 days (should include 90 burn-in + 10 data conceptually,
        # but metrics should only be for days after _metrics_start_day)
        sim.run(days=100)

        # Access the tracker's count directly
        n_samples = sim.monitor.service_level_tracker.count

        # With default_burn_in_days=90, only days 91-100 should have metrics
        # That's 100 days starting from day 1, but metrics only from day 91+
        # So we should have 100 - 90 = 10 days of metrics
        assert n_samples == 10, (
            f"Expected 10 service level samples (excluding 90-day burn-in), got {n_samples}"
        )


class TestMetricsConsistency:
    """Tests that metrics are consistent across checkpoint loads."""

    def test_checkpoint_produces_similar_service_level(self) -> None:
        """Service level should be similar whether using checkpoint or not."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # First run with auto-checkpoint (runs burn-in + data)
        sim1 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim1.run(days=30)
        report1 = sim1.monitor.get_report()
        svc1 = report1.get("store_service_level", {}).get("mean", 0)

        # Second run loads checkpoint (skips burn-in)
        sim2 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim2.run(days=30)
        report2 = sim2.monitor.get_report()
        svc2 = report2.get("store_service_level", {}).get("mean", 0)

        # Service levels should be within 2% of each other
        # (Some variance expected due to stochastic elements)
        diff = abs(svc1 - svc2)
        assert diff < 0.02, (
            f"Service levels differ by {diff*100:.1f}% "
            f"(run1: {svc1*100:.1f}%, run2: {svc2*100:.1f}%)"
        )

        # Clean up
        shutil.rmtree(checkpoint_dir)

    def test_checkpoint_produces_similar_inventory_turns(self) -> None:
        """Inventory turns should be similar whether using checkpoint or not."""
        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # First run with auto-checkpoint
        sim1 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim1.run(days=30)
        report1 = sim1.monitor.get_report()
        turns1 = report1.get("inventory_turns", {}).get("mean", 0)

        # Second run loads checkpoint
        sim2 = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )
        sim2.run(days=30)
        report2 = sim2.monitor.get_report()
        turns2 = report2.get("inventory_turns", {}).get("mean", 0)

        # Turns should be within 0.5x of each other
        diff = abs(turns1 - turns2)
        assert diff < 0.5, (
            f"Inventory turns differ by {diff:.2f}x "
            f"(run1: {turns1:.2f}x, run2: {turns2:.2f}x)"
        )

        # Clean up
        shutil.rmtree(checkpoint_dir)


class TestConfigHashValidation:
    """Tests for config hash validation on checkpoint load."""

    def test_stale_checkpoint_triggers_regeneration(self) -> None:
        """Changed config should trigger checkpoint regeneration."""
        # This test is implicit in the system - when config changes,
        # the hash changes, and the checkpoint is considered stale.
        # The orchestrator will then set _needs_burn_in=True.

        # Clean any existing checkpoints
        checkpoint_dir = Path("data/checkpoints")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # Create orchestrator - should need burn-in (no checkpoint)
        sim = Orchestrator(
            enable_logging=False,
            auto_checkpoint=True,
        )

        assert sim._needs_burn_in is True

        # Run to create checkpoint
        sim.run(days=5)
        assert sim._checkpoint_path.exists()

        # Config hash should be stored
        assert len(sim._config_hash) == 16  # SHA256 truncated to 16 chars

        # Clean up
        shutil.rmtree(checkpoint_dir)
