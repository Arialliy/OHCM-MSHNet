STOPPED_BRANCHES = {
    "PFRMSHNet": "Stopped after Full gate failure.",
    "ERDMSHNet": "Stopped after HC-Val / Full reliability failure.",
    "ERDMSHNetV3": "Stopped after Full gate failure.",
    "CDVMSHNet": "Stopped after Gate-B flat-artifact failure.",
    "ECDVMSHNet": "Stopped after Gate-B flat-artifact failure.",
    "MSCVMSHNet": "Stopped after Gate-B candidate/target-top20 failure.",
    "BCVMSHNet": "Stopped after Gate-D2: residual/shape suppressibility insufficient.",
    "OHCMMSHNetFull": "Stopped after full/prototype branch failure.",
}

STOPPED_TWA_VARIANTS = {
    "twa_bn_recalibrated": (
        "Stopped at Gate-TWA-B: BN recalibration did not improve over TWA without BN."
    ),
}


def assert_branch_allowed(model_name: str, allow_stopped_branch: bool = False):
    if model_name in STOPPED_BRANCHES and not allow_stopped_branch:
        reason = STOPPED_BRANCHES[model_name]
        raise RuntimeError(
            f"{model_name} is a stopped diagnostic branch. "
            f"Reason: {reason} "
            "Use --allow_stopped_branch only for diagnostic reproduction."
        )
