from glassbox.nmpc_benchmark import run_nmpc_benchmark


def test_maintained_nmpc_acceptance_suite_passes() -> None:
    report = run_nmpc_benchmark()

    assert report["summary"]["passed"]  # type: ignore[index]
