"""Static contract checks for the CSF Stage 2-only runner."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "run_stage2_only.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "--stage1-checkpoint" in source
    assert "choices=[1, 2, 3]" in source
    assert "do_train_stage1" not in source
    assert "scene_stage2_csf as stage2" in source
    assert "load_state_dict(source_state, strict=True)" in source
    assert "source_stage1_checkpoint_sha256" in source
    assert "stage2_initial_parameters.json" in source
    assert "csf_memory.pt" in source
    assert "csf_memory_audit.json" in source
    assert "final_checkpoint_sha256" in source
    assert "Output exists; never overwrite" in source
    assert "unchanged_official_P16K4" in source
    assert "P16K4 sampler contract differs" in source
    assert "cross_modal_supcon_loss" not in source

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "do_train_stage2"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "csf_memory",
        "csf_weight",
        "csf_temperature",
        "csf_reconstruction_weight",
    }.issubset(keywords)
    print(
        "CSF_STAGE2_ONLY_CONTRACT_OK",
        {
            "stage1_retrained": False,
            "supported_seeds": [1, 2, 3],
            "strict_checkpoint_load": True,
            "memory_artifact": "csf_memory.pt",
        },
    )


if __name__ == "__main__":
    main()
