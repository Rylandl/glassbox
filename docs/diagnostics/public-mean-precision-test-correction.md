# Precision regression ownership assertion correction

The first committed regression attempt at `41e35ab` completed with 24 passing
tests and 12 failures. Every failed test reached its final storage assertion:
all compiled/eager precision transitions, output dtypes, repeated predictions,
memory values and fingerprints had already passed. The assertion required the
owned NumPy arrays to be marked read-only, although the unchanged constructor
at both `ab1cdc5` and `41e35ab` creates owned, writable NumPy64 copies.

This was a test assumption error. Preserve the original result at
`/private/tmp/glassbox-public-mean-precision/artifacts/2026-09-19/public-mean-precision-correction-v1-regression-attempt1`.
The original plan's phrase "immutable NumPy64 arrays" meant inference must not
mutate or replace persisted model state; it incorrectly implied a storage flag
that the model did not previously promise. This explicit correction tests the
actual preservation requirement: array identity, NumPy64 dtype, writeability
flags and the complete fingerprint must match their pre-inference values.

Commit this test-only correction before rerunning the same 36 tests. No model
source, input, numerical tolerance, case, fitting operation or historical result
changes. The concurrently running saved-mean transition and numerical trials
remain bound to their original clean `41e35ab` checkout; their provenance must
not be relabeled. A passing corrected regression does not override the original
finite-difference qualification failure or establish public adoption.
