"""
Tests for unified resource scaling and the parallelism that consumes it.

`detect_resources` is the single SLURM-first source of truth for worker/thread
counts. `prepare_helios_data` and `get_voxel_metrics` now derive their joblib
worker counts from it (replacing ad-hoc `n_jobs=-1`), so these tests pin:

  1. detect_resources honours SLURM_CPUS_PER_TASK before falling back to the
     local machine's CPU count.
  2. The bin-parallel normal/weight computation that get_voxel_metrics feeds the
     resolved worker count into produces identical results regardless of the
     number of workers.
"""
import numpy as np
import psutil
import pytest

from utils import (
    detect_resources,
    Resources,
    compute_normals_weights_from_points_parallel,
)


class TestDetectResources:
    """Unified, SLURM-first resource discovery."""

    def test_returns_sane_resources(self, monkeypatch):
        """A Resources object with usable, positive fields is always returned."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        res = detect_resources()
        assert isinstance(res, Resources)
        assert res.n_workers >= 1
        assert res.threads_per_worker >= 1
        assert res.mem_per_worker_mb >= 256

    def test_fallback_without_slurm(self, monkeypatch):
        """Without SLURM, n_workers derives from the local logical CPU count."""
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        logical = psutil.cpu_count(logical=True) or 1
        res = detect_resources(target_threads_per_worker=2)
        assert res.n_workers == max(1, logical // 2)

    def test_slurm_cpus_take_precedence(self, monkeypatch):
        """SLURM_CPUS_PER_TASK is honoured ahead of the host CPU count."""
        # Pick an allocation guaranteed to exceed the host's logical core count
        # so the SLURM value dominates the layout deterministically.
        machine_logical = psutil.cpu_count(logical=True) or 1
        slurm_cpus = machine_logical + 100
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(slurm_cpus))
        res = detect_resources(target_threads_per_worker=2)
        assert res.n_workers == slurm_cpus // 2

    def test_threads_per_worker_layout(self, monkeypatch):
        """threads_per_worker tracks the requested layout under a SLURM alloc."""
        machine_logical = psutil.cpu_count(logical=True) or 1
        slurm_cpus = machine_logical + 100
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(slurm_cpus))
        res = detect_resources(target_threads_per_worker=4)
        assert res.threads_per_worker == 4
        assert res.n_workers == slurm_cpus // 4


@pytest.mark.slow
class TestParallelNormalsDeterminism:
    """The bin-parallel normal/weight computation must be worker-count agnostic.

    Marked slow: these tests spawn joblib worker processes.
    """

    @staticmethod
    def _synthetic_points(n=200, seed=0):
        # Clustered into a handful of coarse voxels so multiple bins are
        # populated above knn and real PCA normals are exercised.
        rng = np.random.default_rng(seed)
        return rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float64)

    def test_results_independent_of_n_jobs(self):
        """Identical normals/weights whether run on 1 worker or many."""
        pts = self._synthetic_points()
        normals_1, weights_1 = compute_normals_weights_from_points_parallel(
            pts, voxel_size=5.0, knn=6, n_jobs=1
        )
        normals_n, weights_n = compute_normals_weights_from_points_parallel(
            pts, voxel_size=5.0, knn=6, n_jobs=2
        )
        assert normals_1.shape == (len(pts), 3)
        assert weights_1.shape == (len(pts),)
        np.testing.assert_allclose(normals_1, normals_n)
        np.testing.assert_allclose(weights_1, weights_n)

    def test_positive_n_jobs_produces_finite_output(self):
        """A positive n_jobs (as passed from detect_resources) runs cleanly."""
        pts = self._synthetic_points(n=150)
        normals, weights = compute_normals_weights_from_points_parallel(
            pts, voxel_size=5.0, knn=6, n_jobs=3
        )
        assert normals.shape == (len(pts), 3)
        assert weights.shape == (len(pts),)
        assert np.all(np.isfinite(normals))
        assert np.all(np.isfinite(weights))
