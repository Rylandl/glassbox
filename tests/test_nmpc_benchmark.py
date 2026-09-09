import pytest
from _recorded import assert_recorded_close, recorded_result

from glassbox.workflows.benchmarks.nmpc import run_nmpc_benchmark
from glassbox.workflows.record_results import (
    NMPC_ACCEPTANCE_TOLERANCES,
    NMPC_ACCEPTANCE_VOLATILE,
)


@pytest.mark.slow
def test_maintained_nmpc_acceptance_suite_passes() -> None:
    report = run_nmpc_benchmark()

    assert report["summary"]["passed"]  # type: ignore[index]
    assert_recorded_close(
        report,
        recorded_result("nmpc-acceptance-results.json"),
        tolerances=NMPC_ACCEPTANCE_TOLERANCES,
        ignore=NMPC_ACCEPTANCE_VOLATILE,
    )
