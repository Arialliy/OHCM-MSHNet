#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class FinalStopStateError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FinalStopStateError(f"Missing required JSON file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise FinalStopStateError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FinalStopStateError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def read_text(path: Path) -> str:
    if not path.exists():
        raise FinalStopStateError(f"Missing required text file: {path}")
    return path.read_text(encoding="utf-8")


def first_n_lines(text: str, n: int = 160) -> str:
    return "\n".join(text.splitlines()[:n])


def normalize_token(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def contains_token(text: str, token: str) -> bool:
    return token in text or normalize_token(token) in normalize_token(text)


def collect_numeric_values(obj: Any, key_names: Iterable[str]) -> List[float]:
    keys = set(key_names)
    values: List[float] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) in keys:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
            values.extend(collect_numeric_values(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(collect_numeric_values(item, keys))
    return values


def ensure_no_active_gate_in_top_readme(readme_text: str, errors: List[str]) -> None:
    top = first_n_lines(readme_text, 180)
    suspicious_phrases = [
        "Current active candidate",
        "Next allowed gate: Gate-TWA",
        "Next allowed gate: Gate-TCSR",
        "Next allowed gate: Gate-TCE",
        "PROCEED_TCE4_TO_F3",
        "Gate-TCE-F3-blind-external-once",
        "Gate-LS-B-ep300",
    ]
    allowed_context = [
        "No active AAAI main-method branch remains",
        "no active AAAI main-method branch remains",
        "No active next_allowed_gate",
    ]
    if any(contains_token(top, ctx) for ctx in allowed_context):
        return
    for phrase in suspicious_phrases:
        if contains_token(top, phrase):
            errors.append(
                f"README top block still looks active because it contains {phrase!r}. "
                "Move historical gate text below an archive section and make the top state final-stopped."
            )


def check_required_text_tokens(label: str, text: str, tokens: Iterable[str], errors: List[str]) -> None:
    for token in tokens:
        if not contains_token(text, token):
            errors.append(f"{label} does not contain required token: {token}")


def check_final_stop_state(
    root: Path,
    f3_fail_summary_path: Optional[Path] = None,
    f3_once_lock_path: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    stopped_summary_path: Optional[Path] = None,
    f3_final_report_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = root.resolve()
    tce_dir = root / "docs" / "internal" / "tce_final"

    f3_fail_summary_path = f3_fail_summary_path or (tce_dir / "gate_tce_f3_fail_summary.json")
    f3_once_lock_path = f3_once_lock_path or (tce_dir / "gate_tce_f3_once_lock.json")
    readme_path = readme_path or (root / "README.md")
    stopped_summary_path = stopped_summary_path or (root / "STOPPED_BRANCHES_SUMMARY.md")
    f3_final_report_path = f3_final_report_path or (tce_dir / "gate_tce_f3_blind_external_report.json")
    output_path = output_path or (root / "docs" / "internal" / "final_stop_state_summary.json")

    errors: List[str] = []
    warnings: List[str] = []

    fail_summary = load_json(f3_fail_summary_path)
    once_lock = load_json(f3_once_lock_path)
    readme_text = read_text(readme_path)
    stopped_text = read_text(stopped_summary_path)

    decision = fail_summary.get("decision")
    if decision != "F3_FAIL_NO_REDESIGN":
        errors.append(
            "gate_tce_f3_fail_summary.json must have "
            "decision == 'F3_FAIL_NO_REDESIGN', got "
            f"{decision!r}"
        )

    lock_status = once_lock.get("status")
    if lock_status != "STOPPED_BY_F3_PD_REGRESSION":
        errors.append(
            "gate_tce_f3_once_lock.json must have "
            "status == 'STOPPED_BY_F3_PD_REGRESSION', got "
            f"{lock_status!r}"
        )

    pd_values = collect_numeric_values(fail_summary.get("failed_splits", fail_summary), [
        "min_delta_Pd",
        "delta_Pd",
    ])
    negative_pd_values = [x for x in pd_values if x < 0]
    if not negative_pd_values:
        errors.append(
            "F3 fail summary must contain at least one negative min_delta_Pd / delta_Pd "
            "inside failed_splits."
        )

    not_completed = fail_summary.get("not_completed_splits", {})
    if isinstance(not_completed, Mapping) and "external_sirst3" in not_completed:
        sirst3 = not_completed["external_sirst3"]
        reason_text = json.dumps(sirst3, ensure_ascii=False).lower()
        if "manifest" not in reason_text and "integrity" not in reason_text and "missing" not in reason_text:
            warnings.append(
                "external_sirst3 is listed as not completed, but the reason does not clearly "
                "mention manifest/integrity/missing data."
            )

    if f3_final_report_path.exists():
        errors.append(
            "F3 final report exists even though F3 stopped before final report generation: "
            f"{f3_final_report_path}. Do not use it as official final report."
        )

    check_required_text_tokens(
        "README.md",
        first_n_lines(readme_text, 200),
        [
            "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
            "STOP_TCSR_AT_BANK_AUDIT",
            "No active AAAI main-method branch remains",
        ],
        errors,
    )
    ensure_no_active_gate_in_top_readme(readme_text, errors)

    check_required_text_tokens(
        "STOPPED_BRANCHES_SUMMARY.md",
        stopped_text,
        [
            "F3_FAIL_NO_REDESIGN",
            "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
            "Pd regression",
            "STOP_TCSR_AT_BANK_AUDIT",
            "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
        ],
        errors,
    )

    forbidden_actions = fail_summary.get("forbidden_next_actions", [])
    forbidden_text = json.dumps(forbidden_actions, ensure_ascii=False)
    for token in [
        "threshold search",
        "seed search",
        "checkpoint search",
        "split redefinition",
        "new model training",
    ]:
        if not contains_token(forbidden_text, token):
            warnings.append(f"F3 fail summary forbidden_next_actions does not explicitly list: {token}")

    result: Dict[str, Any] = {
        "gate": "Gate-FINAL-STOP-CONSISTENCY",
        "gate_pass": len(errors) == 0,
        "decision": "READ_ONLY_FAILURE_ARCHIVE_STATE" if len(errors) == 0 else "FINAL_STOP_STATE_INCONSISTENT",
        "root": str(root),
        "checked_files": {
            "f3_fail_summary": str(f3_fail_summary_path),
            "f3_once_lock": str(f3_once_lock_path),
            "readme": str(readme_path),
            "stopped_summary": str(stopped_summary_path),
            "f3_final_report": str(f3_final_report_path),
        },
        "required_status": {
            "f3_fail_decision": "F3_FAIL_NO_REDESIGN",
            "f3_once_lock_status": "STOPPED_BY_F3_PD_REGRESSION",
            "top_readme_state": "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
        },
        "observed": {
            "f3_fail_decision": decision,
            "f3_once_lock_status": lock_status,
            "negative_pd_values": negative_pd_values,
            "f3_final_report_exists": f3_final_report_path.exists(),
        },
        "errors": errors,
        "warnings": warnings,
        "forbidden_next_actions": [
            "new evaluation",
            "new training",
            "threshold search",
            "seed search",
            "checkpoint search",
            "split redefinition",
            "SIRST3 mask imputation",
            "TCSR Stage 2",
            "new verifier",
            "new suppression head",
            "loss.py modification",
            "net.py modification",
            "train.py modification",
            "dataset.py modification",
        ],
        "allowed_next_actions": [
            "archive final stopped state",
            "generate failure-analysis table",
            "prepare internal report",
            "freeze repository",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if errors:
        raise FinalStopStateError(
            "Gate-FINAL-STOP-CONSISTENCY failed:\n" + "\n".join(f"- {e}" for e in errors)
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check final stopped state after TCE-4 F3 external Pd regression."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--f3_fail_summary", type=Path, default=None)
    parser.add_argument("--f3_once_lock", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--stopped_summary", type=Path, default=None)
    parser.add_argument("--f3_final_report", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: docs/internal/final_stop_state_summary.json",
    )
    args = parser.parse_args()

    try:
        result = check_final_stop_state(
            root=args.root,
            f3_fail_summary_path=args.f3_fail_summary,
            f3_once_lock_path=args.f3_once_lock,
            readme_path=args.readme,
            stopped_summary_path=args.stopped_summary,
            f3_final_report_path=args.f3_final_report,
            output_path=args.output,
        )
    except FinalStopStateError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
