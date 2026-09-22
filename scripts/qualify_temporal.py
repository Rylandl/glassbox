"""Reuse the matched offline fit/physical-query evaluator for temporal compression."""

import qualify_accumulator as qualification
from run_dart import ROOT

qualification.PROTOCOL = ROOT / "docs/harness/nonlinear-temporal-offline-v1.json"

if __name__ == "__main__":
    qualification.main()
