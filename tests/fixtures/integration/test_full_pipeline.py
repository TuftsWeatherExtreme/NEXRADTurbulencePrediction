# test_full_pipeline.py
# Integration test for the full NEXRAD turbulence prediction pipeline:
#   CSV gen -> model inputs -> dataloader -> train/test
#
# Designed to run on the Tufts HPC cluster (SLURM).
# Uses minimal synthetic fixtures to avoid S3/real data dependencies.
#
# Usage:
#   pytest tests/test_full_pipeline.py -m integration -s
#
# Prerequisites:
#   - $REPO_PATH env var is set
#   - Conda environment is activated (nexrad_env)
#   - pytest, torch, xarray, numpy are available in the environment

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("REPO_PATH", Path(__file__).resolve().parents[2]))
TEST_OUTPUT_DIR = REPO_ROOT / "tests" / "tmp" / "pipeline_integration"
FIXTURE_COMPRESSED_DIR = TEST_OUTPUT_DIR / "model_inputs" / "compressed"
DATALOADER_OUTPUT_PTH = TEST_OUTPUT_DIR / "test_dataloader.pth"
TRAINED_MODEL_OUTPUT_DIR = TEST_OUTPUT_DIR / "trained_model_outputs"
DECOMPRESSED_DIR = REPO_ROOT / "decompressed"

# SLURM polling settings
SLURM_POLL_INTERVAL_SECONDS = 30
SLURM_TIMEOUT_SECONDS = 60 * 60 * 2  # 2 hours max per stage


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_synthetic_netcdf(path: Path, turb_level: int = 2) -> None:
    """
    Creates a minimal synthetic NetCDF file that matches the schema expected
    by RadarDataLoader:
      - attrs: lat, lon, alt, turb  (turb is the label, must be last)
      - data var: reflectivity of shape (16, 16, 10)
    """
    reflectivity = np.full((16, 16, 10), np.nan, dtype=np.float32)
    reflectivity[8, 8, 5] = 20.0  # one real value so the grid isn't entirely NaN

    ds = xr.Dataset(
        {"reflectivity": (["x", "y", "z"], reflectivity)},
        attrs={
            "lat": 42.36,
            "lon": -71.05,
            "alt": 10000.0,
            "turb": float(turb_level),
        },
    )
    ds.to_netcdf(path)


def _make_synthetic_tar_xz(tar_path: Path, n_files: int = 5) -> None:
    """
    Creates a .tar.xz archive containing n_files synthetic NetCDF files,
    mirroring the structure produced by generate_model_inputs.sh:
        compressed/001.tar.xz  ->  decompressed/001/<file>.nc
    """
    stem = tar_path.stem.replace(".tar", "")  # e.g. "001"
    staging_dir = tar_path.parent / f"_staging_{stem}"
    inner_dir = staging_dir / stem
    inner_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        nc_path = inner_dir / f"input_{i:04d}.nc"
        _make_synthetic_netcdf(nc_path, turb_level=i % 5)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:xz") as tar:
        tar.add(staging_dir / stem, arcname=stem)

    shutil.rmtree(staging_dir)


def _build_synthetic_compressed_fixtures(n_archives: int = 3, files_per_archive: int = 5) -> None:
    """
    Populates FIXTURE_COMPRESSED_DIR with n_archives synthetic .tar.xz files.
    Skips creation if fixtures already exist (idempotent).
    """
    FIXTURE_COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_archives + 1):
        tar_path = FIXTURE_COMPRESSED_DIR / f"{i:03d}.tar.xz"
        if not tar_path.exists():
            _make_synthetic_tar_xz(tar_path, n_files=files_per_archive)
            print(f"Created fixture: {tar_path}")
        else:
            print(f"Fixture already exists, skipping: {tar_path}")


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def _submit_job(script: Path, args: list[str] | None = None) -> str:
    """
    Submits a SLURM job and returns the job ID string.
    Raises if submission fails.
    """
    cmd = ["sbatch", "--parsable", str(script)]
    if args:
        cmd += args

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"sbatch submission failed for {script.name}\n"
        f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    job_id = result.stdout.strip().split(";")[0]  # --parsable gives "jobid;cluster"
    print(f"Submitted {script.name} -> job {job_id}")
    return job_id


def _wait_for_job(job_id: str, timeout: int = SLURM_TIMEOUT_SECONDS) -> str:
    """
    Polls sacct until the job reaches a terminal state.
    Returns the final SLURM state string (COMPLETED, FAILED, etc.).
    Raises TimeoutError if the job doesn't finish within `timeout` seconds.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            check=False,
        )
        states = [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
        if states:
            main_state = states[0].split("+")[0]  # strip trailing flags
            if main_state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                print(f"Job {job_id} finished with state: {main_state}")
                return main_state
        print(f"Job {job_id} state: {states} — waiting {SLURM_POLL_INTERVAL_SECONDS}s...")
        time.sleep(SLURM_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_test_dirs():
    """Creates output dirs and synthetic fixtures once per test module."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _build_synthetic_compressed_fixtures(n_archives=3, files_per_archive=5)
    yield
    # Teardown: remove test outputs after run.
    # Comment out the line below to inspect artifacts after a run.
    # shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_decompressed():
    """Ensures the decompressed/ scratch dir is clean before and after each test."""
    if DECOMPRESSED_DIR.exists():
        shutil.rmtree(DECOMPRESSED_DIR)
    yield
    if DECOMPRESSED_DIR.exists():
        shutil.rmtree(DECOMPRESSED_DIR)


# ---------------------------------------------------------------------------
# Stage 1 — Dataloader generation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_stage_dataloader_generation():
    """
    Submits generate_dataloader.sh against synthetic compressed fixtures
    and asserts:
      - SLURM job completes successfully
      - .pth output file exists and is non-empty
      - torch.load can deserialize it
      - decompressed/ scratch directory is cleaned up
    """
    if DATALOADER_OUTPUT_PTH.exists():
        DATALOADER_OUTPUT_PTH.unlink()

    script = REPO_ROOT / "hpc_scripts" / "data_processing" / "generate_dataloader.sh"

    # generate_dataloader.sh hardcodes model_inputs/compressed relative to REPO_PATH.
    # We temporarily symlink our fixture dir there so the script picks up test data.
    real_compressed = REPO_ROOT / "model_inputs" / "compressed"
    symlink_created = False
    if not real_compressed.exists():
        real_compressed.parent.mkdir(parents=True, exist_ok=True)
        real_compressed.symlink_to(FIXTURE_COMPRESSED_DIR)
        symlink_created = True

    try:
        job_id = _submit_job(script, args=[str(DATALOADER_OUTPUT_PTH)])
        state = _wait_for_job(job_id)

        assert state == "COMPLETED", (
            f"generate_dataloader.sh job {job_id} ended in state '{state}'.\n"
            f"Inspect with: sacct -j {job_id} --format=JobID,State,ExitCode"
        )
    finally:
        if symlink_created and real_compressed.is_symlink():
            real_compressed.unlink()

    assert DATALOADER_OUTPUT_PTH.exists(), (
        "Expected dataloader .pth was not created."
    )
    assert DATALOADER_OUTPUT_PTH.stat().st_size > 0, (
        "Dataloader .pth file is empty."
    )

    loaded = torch.load(DATALOADER_OUTPUT_PTH, map_location="cpu", weights_only=False)
    assert loaded is not None, "torch.load returned None."
    assert len(loaded) > 0, (
        "Loaded dataloader has 0 entries. "
        "Check that synthetic .tar.xz fixtures were read correctly."
    )

    assert not DECOMPRESSED_DIR.exists(), (
        "Temporary decompressed/ directory was not cleaned up by generate_dataloader.sh."
    )


# ---------------------------------------------------------------------------
# Stage 2 — Model training and evaluation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_stage_train_and_test_model():
    """
    Submits train_and_test_model.sh against the dataloader produced in Stage 1
    and asserts:
      - SLURM job completes successfully
      - A .pth model weights file is saved to trained_model_outputs/
      - A _results.txt file is saved alongside it
      - Results file contains all expected output fields
    """
    assert DATALOADER_OUTPUT_PTH.exists(), (
        "Dataloader .pth not found. "
        "Run test_stage_dataloader_generation first, or run both tests together."
    )

    script = REPO_ROOT / "hpc_scripts" / "model_training" / "train_and_test_model.sh"

    # train_and_test_model.py hardcodes DATALOADER_PATH and OUTPUT_DIR.
    # We override both via environment variables so the job uses test paths
    # without modifying production code.
    env_overrides = {
        **os.environ,
        "TEST_DATALOADER_PATH": str(DATALOADER_OUTPUT_PTH),
        "TEST_OUTPUT_DIR": str(TRAINED_MODEL_OUTPUT_DIR),
        "TEST_EPOCHS": "1",       # 1 epoch is enough to verify the pipeline runs
        "TEST_MODE": "1",
    }

    result = subprocess.run(
        ["sbatch", "--parsable", str(script), "hybrid", "mse", "42"],
        capture_output=True,
        text=True,
        check=False,
        env=env_overrides,
    )
    assert result.returncode == 0, (
        f"sbatch submission failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    job_id = result.stdout.strip().split(";")[0]
    print(f"Submitted train_and_test_model.sh -> job {job_id}")

    state = _wait_for_job(job_id)
    assert state == "COMPLETED", (
        f"train_and_test_model.sh job {job_id} ended in state '{state}'.\n"
        f"Inspect with: sacct -j {job_id} --format=JobID,State,ExitCode"
    )

    model_files = list(TRAINED_MODEL_OUTPUT_DIR.glob("*_best_hybrid_mse_model_w_seed_42.pth"))
    results_files = list(TRAINED_MODEL_OUTPUT_DIR.glob("*_best_hybrid_mse_model_w_seed_42_results.txt"))

    assert len(model_files) >= 1, (
        f"No model .pth found in {TRAINED_MODEL_OUTPUT_DIR}. "
        "Training may have failed silently."
    )
    assert len(results_files) >= 1, (
        f"No results .txt found in {TRAINED_MODEL_OUTPUT_DIR}."
    )

    results_text = results_files[-1].read_text()  # most recent if multiple runs
    for expected_field in [
        "Accuracy is:",
        "false positive rate",
        "false negative rate",
        "actual distribution of classes",
        "model's distribution of classes",
    ]:
        assert expected_field.lower() in results_text.lower(), (
            f"Expected field '{expected_field}' not found in results file.\n"
            f"Results file contents:\n{results_text}"
        )


# ---------------------------------------------------------------------------
# Stage 3 — Dataloader content integrity (no SLURM, runs inline)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_stage_dataloader_content_integrity():
    """
    Loads the generated dataloader directly (no SLURM) and validates that
    each item has the correct tensor shape and a valid integer label in [0, 9].

    This catches:
      - NaN-only feature tensors making it through nan_to_num
      - Label encoding mismatches
      - Shape regressions in RadarDataLoader
    """
    assert DATALOADER_OUTPUT_PTH.exists(), (
        "Dataloader .pth not found. Run test_stage_dataloader_generation first."
    )

    dataset = torch.load(DATALOADER_OUTPUT_PTH, map_location="cpu", weights_only=False)
    assert len(dataset) > 0, "Dataset is empty."

    features_0, label_0 = dataset[0]

    # 3 scalar attrs (lat, lon, alt) + 16*16*10 reflectivity grid cells = 2563
    expected_feature_len = 3 + (16 * 16 * 10)
    assert features_0.shape == torch.Size([expected_feature_len]), (
        f"Unexpected feature vector shape: {features_0.shape}. "
        f"Expected ({expected_feature_len},)."
    )

    assert isinstance(label_0, int), (
        f"Label should be an int, got {type(label_0)}."
    )
    assert 0 <= label_0 <= 9, (
        f"Label {label_0} is outside the expected turbulence class range [0, 9]."
    )

    # RadarDataLoader replaces NaNs with -32.0 — no NaNs should remain
    assert not torch.isnan(features_0).any(), (
        "Feature tensor still contains NaN values after nan_to_num replacement."
    )