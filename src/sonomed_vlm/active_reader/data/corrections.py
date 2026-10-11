"""Versioned, idempotent author corrections; source data is never written."""

from copy import deepcopy

NOTICE = "https://github.com/jhl-Det/CVA-Net#news"
VERSION = "cva-net-2023-03-25-v1"
EXCLUDED = "x66ef02e7f1b9a0ef"
COUNTERPART = "x3b88488853e8b7d1"
SHARED = {"x63c9ba1377f35bf6", "x5a1c46ec6377e946", "2390fbea047347b", "7a39ab5d4970bf89"}


def apply_corrections(cases):
    output, changes = [], []
    for original in cases:
        case = deepcopy(original)
        source_id = case.source_relative_path.replace("\\", "/").rstrip("/").split("/")[-1]
        if source_id == EXCLUDED:
            changes.append(
                {"source_id": source_id, "action": "exclude_duplicate", "counterpart": COUNTERPART}
            )
            continue
        if source_id in SHARED | {COUNTERPART}:
            if case.label != "malignant":
                changes.append({"source_id": source_id, "action": "correct_to_malignant"})
            case.label = "malignant"
            case.label_provenance = VERSION
            tag = VERSION + (":shared_patient" if source_id in SHARED else ":duplicate_counterpart")
            case.applied_corrections = sorted(set(case.applied_corrections) | {tag})
            if source_id in SHARED:
                case.group_id = "author_shared_patient_20230325"
                case.group_basis = "author_notice"
        output.append(case)
    return output, {
        "version": VERSION,
        "source": NOTICE,
        "changes": changes,
        "excluded_source_present": any(
            c.source_relative_path.split("/")[-1] == EXCLUDED for c in cases
        ),
    }
