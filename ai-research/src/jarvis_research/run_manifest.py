"""HZN-IR-11 D1 — run-manifest validation for resumable experiment runners.

A runner that resumes from a partial results file must prove the saved
cells were produced under the SAME configuration it is about to run —
otherwise stale results are silently reused (IR-10 review finding).
`validate_resume_manifest` compares the stored manifest against the
current configuration and returns a list of mismatches; runners must
refuse to reuse cells when the list is non-empty.

Compared keys (when present in the current config): model identity,
model revision, harness version, catalog fingerprints, seed list,
size/condition lists, prompt template hashes, instance counts. Keys
missing from the STORED manifest count as mismatches (the run predates
the binding requirement); extra keys in storage do not.
"""

REQUIRED_KEYS = ("model_revision", "harness_version")


def validate_resume_manifest(stored: dict, current: dict) -> list:
    """Return a list of human-readable mismatch descriptions.
    Empty list = safe to reuse stored cells."""
    errors = []
    if not isinstance(stored, dict) or not stored:
        return ["no stored manifest — run predates binding requirement"]
    for key in REQUIRED_KEYS:
        if key not in stored:
            errors.append(f"stored manifest missing required key: {key}")
    for key, cur_val in current.items():
        if key not in stored:
            errors.append(f"stored manifest missing key: {key}")
        elif stored[key] != cur_val:
            errors.append(
                f"manifest mismatch on {key}: "
                f"stored={str(stored[key])[:60]!r} "
                f"current={str(cur_val)[:60]!r}")
    return errors


def resume_or_raise(stored: dict, current: dict) -> None:
    """Hard gate: raise RuntimeError unless reuse is safe."""
    errors = validate_resume_manifest(stored, current)
    if errors:
        raise RuntimeError(
            "refusing to resume from mismatched run manifest:\n  - "
            + "\n  - ".join(errors))
