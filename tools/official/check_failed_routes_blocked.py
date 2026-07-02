#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


FAILED_ROUTES = {
    "sps": "Gate0 / TC-v2 / region / peak failed.",
    "erd": "ERD-v2 HC-Val failed.",
    "erd_v3": "ERD-v3 Full failed.",
    "tcd": "TCE soft-label information audit failed.",
    "pfr": "PFR evidence branch contaminated and FP components exploded.",
    "apf": "APF Gate-A candidate audit failed: flat background candidates.",
    "eacf": "EACF-v1 stopped after Gate-F3 identity collapse.",
    "eacf_v1": "EACF-v1 stopped after Gate-F3 identity collapse.",
    "sacf": "SACF-v1 stopped after Gate-S2a identity collapse.",
    "sacf_v1": "SACF-v1 stopped after Gate-S2a identity collapse.",
    "SPSOHEM": "SPS Gate0 failed; training is blocked until a new candidate census passes.",
    "MSHNetSPSOHEM": "SPS Gate0 failed; training is blocked until a new candidate census passes.",
    "ERDMSHNet": "ERD-v2/v3 stopped after gate failures.",
    "ERDMSHNetV3": "ERD-v2/v3 stopped after gate failures.",
    "TCDMSHNet": "TCD stopped because teacher/student soft-label signal was insufficient.",
    "PFRMSHNet": "PFR stopped after Full Gate and head-audit evidence pollution.",
    "APFOHEM": "APF Gate-A candidate audit failed: flat background candidates.",
    "MSHNetAPFOHEM": "APF Gate-A candidate audit failed: flat background candidates.",
    "EACFMSHNet": "EACF-v1 stopped after Gate-F3 identity collapse.",
    "SACFMSHNet": "SACF-v1 stopped after Gate-S2a identity collapse.",
}

EXPECTED_BLOCKED_SCRIPTS = [
    "tools/official/train_sps_seed.sh",
    "tools/official/train_erd_seed.sh",
    "tools/official/train_erd_v3_seed.sh",
    "tools/official/train_tcd_seed.sh",
    "tools/official/train_pfr_seed.sh",
    "tools/official/train_apf_seed.sh",
    "tools/official/train_eacf_seed.sh",
    "tools/official/train_sacf_seed.sh",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Block known failed training routes by default.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--allow_failed_route", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    if args.model_name in FAILED_ROUTES and not args.allow_failed_route:
        print(f"[BLOCKED] {args.model_name}: {FAILED_ROUTES[args.model_name]}")
        print("Use --allow_failed_route only for failure-analysis reruns, never for AAAI decisions.")
        return 2

    if args.model_name in FAILED_ROUTES and args.allow_failed_route:
        record = {
            "model_name": args.model_name,
            "warning": FAILED_ROUTES[args.model_name],
            "reason": args.reason,
        }
        print(json.dumps(record, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
