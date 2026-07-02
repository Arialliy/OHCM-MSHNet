#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FORBIDDEN = {
    "seed_search",
    "checkpoint_search",
    "threshold_search",
    "BN_recalibration_tuning",
    "TCSR_training",
    "new_loss",
    "new_model_structure",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def require(condition: bool, reason: str, failures: List[str]) -> None:
    if not condition:
        failures.append(reason)


def tcsr_stop_present(tcsr: Dict[str, Any]) -> bool:
    return (
        tcsr.get("decision") == "STOP_TCSR_AT_BANK_AUDIT"
        or tcsr.get("next_allowed_gate") == "STOP_TCSR_AT_BANK_AUDIT"
        or tcsr.get("stop_decision") == "STOP_TCSR_AT_BANK_AUDIT"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-TCE-F0 freeze consistency checker.")
    parser.add_argument("--tcsr_gate_a_summary", required=True)
    parser.add_argument("--tce_frozen_plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tcsr = load_json(args.tcsr_gate_a_summary)
    plan = load_json(args.tce_frozen_plan)

    failures: List[str] = []
    require(tcsr.get("gate_pass") is False, "TCSR Gate-A must be failed/stopped", failures)
    require(tcsr_stop_present(tcsr), "TCSR stop decision missing", failures)

    method = plan.get("method", {})
    require(plan.get("decision") == "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE", "wrong freeze decision", failures)
    require(method.get("name") == "TCE-4-OHEM", "method name must be TCE-4-OHEM", failures)
    require(method.get("base") == "MSHNetOHEM", "base must be MSHNetOHEM", failures)
    require(method.get("checkpoints") == [250, 300, 350, 400], "checkpoints must be [250,300,350,400]", failures)
    require(method.get("aggregation") == "existing_official_tce_aggregation", "aggregation must stay frozen", failures)
    require(float(method.get("threshold", -1.0)) == 0.5, "threshold must be fixed at 0.5", failures)
    require(method.get("training") == "no_new_training", "must not require new training", failures)
    require(int(method.get("inference_forward_count", -1)) == 4, "TCE-4 must report 4 forwards", failures)

    forbidden = set(plan.get("forbidden", []))
    for key in sorted(REQUIRED_FORBIDDEN):
        require(key in forbidden, f"forbidden action missing: {key}", failures)

    gate_pass = len(failures) == 0
    result = {
        "gate": "Gate-TCE-F0-freeze-consistency",
        "gate_pass": gate_pass,
        "decision": "PROCEED_TO_TCE_INTERNAL_AGGREGATION" if gate_pass else "STOP_TCE_FINALIZATION_FREEZE_INCONSISTENT",
        "failures": failures,
        "next_allowed_gate": "Gate-TCE-F1-internal-evidence-aggregation" if gate_pass else "STOP",
        "frozen_method": method,
        "forbidden": sorted(forbidden),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not gate_pass:
        raise SystemExit("Gate-TCE-F0 failed. Fix status/freeze inconsistency only.")


if __name__ == "__main__":
    main()
