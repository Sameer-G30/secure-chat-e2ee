"""Exercise the TF-IDF baseline OFAT catalog without fitting the 71k corpus."""

# Import importlib so tests can load the sweep CLI without putting scripts/ on PYTHONPATH.
import importlib.util

# Import Path to resolve the sweep script relative to this test file.
from pathlib import Path

# Import the published grids so catalog assertions stay anchored to baseline.py.
from secure_chat_ml.baseline import DEFAULT_C_GRID, EXPANDED_THRESHOLD_GRID, WIDENED_C_GRID


# Load scripts/sweep_baseline_params.py as a module for catalog helpers.
def _load_sweep_module():
    """Return the sweep module loaded from ml/scripts/sweep_baseline_params.py."""

    # Resolve the script path from tests/ -> ml/scripts/.
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_baseline_params.py"
    # Build a spec from the file location so import machinery can exec it.
    spec = importlib.util.spec_from_file_location("sweep_baseline_params", script_path)
    # Fail loudly if the file is missing rather than returning a dummy catalog.
    assert spec is not None and spec.loader is not None
    # Allocate an empty module object for the loader to populate.
    module = importlib.util.module_from_spec(spec)
    # Execute the sweep script (imports baseline constants; does not train).
    spec.loader.exec_module(module)
    # Return the populated module for catalog assertions.
    return module


# Confirm the catalog starts with the expanded-grid retrain of the published recipe.
def test_ofat_catalog_starts_with_expanded_grid_baseline() -> None:
    """Assert run 00 changes only the VAL grids, not a training knob."""

    # Load the sweep module once for this assertion.
    sweep = _load_sweep_module()
    # Build the numbered catalog the CLI would print on --dry-run.
    runs = sweep.build_ofat_runs()
    # The first job must exist so folder 00_* is always the expanded-grid baseline.
    assert runs, "OFAT catalog must not be empty"
    # Read the first catalog row.
    first = runs[0]
    # Folder names start at 00 so filesystem sort matches training order.
    assert str(first["run_id"]).startswith("00_")
    # The first run is tagged as a grid expansion, not a TF-IDF knob change.
    assert first["changed_parameter"] == "val_grids"
    # Training knobs on run 00 must match the published recipe exactly.
    assert first["values"] == sweep.BASELINE_VALUES


# Confirm every later run changes exactly one logical group.
def test_ofat_catalog_changes_one_group_per_run() -> None:
    """Assert each variant differs from the published knobs in a single group."""

    # Load the sweep module for catalog helpers.
    sweep = _load_sweep_module()
    # Walk every catalog row after the expanded-grid baseline.
    for run in sweep.build_ofat_runs()[1:]:
        # Copy published knobs so we can apply the same mutation the catalog used.
        expected = dict(sweep.BASELINE_VALUES)
        # Read which group this run changes.
        parameter = run["changed_parameter"]
        # Read the alternative value stored on the catalog row.
        alternative = run["changed_value"]
        # ngram_range is one group that updates two integer keys together.
        if parameter == "ngram_range":
            # Catalog stores the pair as a tuple (min, max).
            low, high = alternative
            # Unigrams-only / trigrams / bigrams-only must set both bounds.
            expected["ngram_min"] = int(low)
            # Upper bound is inclusive.
            expected["ngram_max"] = int(high)
        else:
            # Scalar groups overwrite exactly one key.
            expected[parameter] = alternative
        # The catalog values dict must match the one-group mutation and nothing else.
        assert run["values"] == expected, run["run_id"]
        # Count how many published keys actually differ for extra safety.
        differing = [
            key
            for key, value in sweep.BASELINE_VALUES.items()
            if run["values"][key] != value
        ]
        # ngram_range may change one or both bounds; no other group may move.
        if parameter == "ngram_range":
            # The changed keys must be a non-empty subset of the n-gram pair.
            assert set(differing).issubset({"ngram_min", "ngram_max"})
            # At least one bound must differ or this is not an OFAT alternative.
            assert differing
            # Every other published key must still match the baseline recipe.
            assert run["values"]["ngram_min"] == alternative[0]
            # Upper bound is inclusive and must match the catalog pair.
            assert run["values"]["ngram_max"] == alternative[1]
        else:
            # A scalar OFAT run must not leak a second knob change.
            assert differing == [parameter]


# Confirm the catalog size is 1 + the number of listed alternatives.
def test_ofat_catalog_size_matches_the_alternative_lists() -> None:
    """Assert we did not silently drop or duplicate an OFAT alternative."""

    # Load the sweep module for OFAT_ALTERNATIVES.
    sweep = _load_sweep_module()
    # Count every alternative across every group.
    n_variants = sum(len(values) for values in sweep.OFAT_ALTERNATIVES.values())
    # Plus the expanded-grid baseline that changes no training knob.
    expected = 1 + n_variants
    # The numbered catalog must match that count exactly.
    assert len(sweep.build_ofat_runs()) == expected
    # Guard against an accidentally empty alternatives dict.
    assert n_variants >= 10


# Confirm run ids are unique and zero-padded.
def test_ofat_run_ids_are_unique_and_zero_padded() -> None:
    """Assert folder names sort in training order and never collide."""

    # Load the sweep module for the catalog.
    sweep = _load_sweep_module()
    # Collect every folder name.
    run_ids = [str(run["run_id"]) for run in sweep.build_ofat_runs()]
    # Colliding ids would overwrite reports.
    assert len(run_ids) == len(set(run_ids))
    # The numeric prefix must match catalog order (00, 01, 02, ...).
    prefixes = [run_id.split("_", 1)[0] for run_id in run_ids]
    # Zero-pad to two digits so 10 does not sort before 9.
    expected = [f"{index:02d}" for index in range(len(run_ids))]
    # Prefixes must equal the zero-padded index sequence.
    assert prefixes == expected


# Confirm the sweep module reuses the shared expanded/widened grids.
def test_sweep_grids_match_baseline_module_constants() -> None:
    """Assert the CLI cannot drift from EXPANDED_THRESHOLD_GRID / WIDENED_C_GRID."""

    # Load the sweep module's imported constants.
    sweep = _load_sweep_module()
    # Threshold grid must include 0.20 and 0.25 via the shared constant.
    assert sweep.EXPANDED_THRESHOLD_GRID == EXPANDED_THRESHOLD_GRID
    # C grid must be the widened set, not the published three-value grid.
    assert sweep.WIDENED_C_GRID == WIDENED_C_GRID
    # The published three-value grid must remain a strict subset.
    assert set(DEFAULT_C_GRID).issubset(set(sweep.WIDENED_C_GRID))


# Confirm ranking prefers the DistilBERT-README combined TEST mean.
def test_rank_rows_orders_by_combined_mean_then_scam_recall() -> None:
    """Assert a higher combined mean beats a higher raw scam recall."""

    # Load ranking helpers from the sweep module.
    sweep = _load_sweep_module()
    # Build two synthetic ranking rows with TEST metrics only.
    lower_mean = {
        "run_id": "low_mean_high_recall",
        "test_scam_recall": 0.99,
        "test_legit_precision": 0.80,
        "test_legit_recall": 0.90,
        "test_accuracy": 0.85,
        "combined_mean": (0.99 + 0.80 + 0.85) / 3.0,
        "chat_scam_recall": 1.0,
        "chat_legit_recall": 1.0,
        "source": "sweep_retrain",
        "floor_feasible": True,
    }
    # Combined mean is higher even though scam recall is slightly lower.
    higher_mean = {
        "run_id": "high_mean",
        "test_scam_recall": 0.98,
        "test_legit_precision": 0.95,
        "test_legit_recall": 0.90,
        "test_accuracy": 0.96,
        "combined_mean": (0.98 + 0.95 + 0.96) / 3.0,
        "chat_scam_recall": 0.5,
        "chat_legit_recall": 0.5,
        "source": "sweep_retrain",
        "floor_feasible": True,
    }
    # Rank with the same key the CLI uses for best_combination.
    ranked = sweep._rank_rows([lower_mean, higher_mean])
    # The higher combined-mean row must win, matching the DistilBERT README rule.
    assert ranked[0]["run_id"] == "high_mean"
    # _pick_best must skip nothing here because both rows are retrains with a feasible floor.
    best = sweep._pick_best(ranked)
    # The winner must be the high-mean retrain.
    assert best is not None
    # Confirm the run_id rather than object identity.
    assert best["run_id"] == "high_mean"


# Confirm _pick_best skips the published original-grid reference row.
def test_pick_best_skips_published_original_grid_reference() -> None:
    """Assert 'best combination' is a retrained sweep run, not the published baseline."""

    # Load ranking helpers from the sweep module.
    sweep = _load_sweep_module()
    # A published reference that would otherwise win on combined mean.
    published = {
        "run_id": "published_baseline_original_grids",
        "source": "published_baseline_original_grids",
        "test_scam_recall": 1.0,
        "test_legit_precision": 1.0,
        "test_legit_recall": 1.0,
        "test_accuracy": 1.0,
        "combined_mean": 1.0,
        "floor_feasible": True,
    }
    # A real OFAT retrain with a lower combined mean.
    retrain = {
        "run_id": "01_max_features_10000",
        "source": "sweep_retrain",
        "test_scam_recall": 0.9,
        "test_legit_precision": 0.9,
        "test_legit_recall": 0.9,
        "test_accuracy": 0.9,
        "combined_mean": 0.9,
        "floor_feasible": True,
    }
    # Rank so the published row is first.
    ranked = sweep._rank_rows([published, retrain])
    # The picker must still return the retrain.
    best = sweep._pick_best(ranked)
    # Guard against an empty catalog edge case.
    assert best is not None
    # The published folder is a comparison reference, not the sweep winner.
    assert best["run_id"] == "01_max_features_10000"
