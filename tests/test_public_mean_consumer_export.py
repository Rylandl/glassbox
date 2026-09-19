"""Consumer-roster selection must not depend on future truth or model outputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.experimental.public_mean_consumer_export import select_queries


def inputs(tmp_path):
    model = SimpleNamespace(
        history_steps=2,
        horizon_steps=1,
        contract={"state_channels": ["x"], "input_channels": ["u"]},
    )
    rows = []
    for scope in ("primary", "wind_shift"):
        for kind in ("factual", "response"):
            for order in (2, 0, 1):
                name = f"{scope}-{kind}-{order}"
                arrays = dict(
                    past_states=np.zeros((3, 1)),
                    past_inputs=np.zeros((2, 1)),
                    future_inputs=np.ones((1, 1)),
                    target=np.full((1, 1), np.nan),
                    valid=np.zeros(1, dtype=bool),
                )
                if kind == "response":
                    arrays["factual_inputs"] = np.zeros((1, 1))
                np.savez_compressed(tmp_path / (name + ".npz"), **arrays)
                rows.append(
                    dict(
                        parent="parent",
                        id=name,
                        scope=scope,
                        kind=kind,
                        path=name + ".npz",
                        history_eligible=True,
                    )
                )
    return model, rows


def test_selects_first_last_inputs_despite_no_future_truth(tmp_path):
    model, rows = inputs(tmp_path)
    chosen = select_queries(rows, tmp_path, model)
    assert len(chosen) == 8
    assert all(row["id"].endswith(("-0", "-2")) for row, _ in chosen)
    assert [(row["parent"], row["id"]) for row, _ in chosen] == sorted(
        (row["parent"], row["id"]) for row, _ in chosen
    )
    assert all("target" not in arrays and "valid" not in arrays for _, arrays in chosen)
    assert sum(2 if row["kind"] == "response" else 1 for row, _ in chosen) == 12


@pytest.mark.parametrize(
    "defect", ["shortage", "unknown_scope", "unknown_kind", "duplicate"]
)
def test_roster_defects_do_not_get_replaced_by_other_strata(tmp_path, defect):
    model, rows = inputs(tmp_path)
    if defect == "shortage":
        rows = [
            r
            for r in rows
            if not (
                r["scope"] == "primary"
                and r["kind"] == "response"
                and not r["id"].endswith("-0")
            )
        ]
    elif defect == "unknown_scope":
        rows[0]["scope"] = "new_shift"
    elif defect == "unknown_kind":
        rows[0]["kind"] = "counterfactual"
    else:
        rows.append(dict(rows[0]))
    with pytest.raises(ValueError):
        select_queries(rows, tmp_path, model)


def test_history_eligibility_and_full_future_input_completeness_are_required(tmp_path):
    model, rows = inputs(tmp_path)
    selected = next(
        r
        for r in rows
        if r["scope"] == "primary"
        and r["kind"] == "response"
        and r["id"].endswith("-0")
    )
    path = tmp_path / selected["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["factual_inputs"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    result = select_queries(rows, tmp_path, model)
    assert selected["id"] not in {row["id"] for row, _ in result}
    remaining = next(
        r
        for r in rows
        if r["scope"] == "primary"
        and r["kind"] == "response"
        and r["id"].endswith("-1")
    )
    remaining["history_eligible"] = False
    with pytest.raises(ValueError, match="insufficient"):
        select_queries(rows, tmp_path, model)
