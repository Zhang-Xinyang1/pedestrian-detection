"""Static integration checks for the isolated CSF Stage 2 loop."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "scene_stage2_csf.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "from prototype_comparison import" in source
    assert "cross_modal_supcon_loss" not in source
    assert "identity_feature = feat[1]" in source
    assert "identity_feature.shape[1] != 768" in source
    assert "prototype_comparison_loss(" in source
    assert "scale_before = float(scaler.get_scale())" in source
    assert "step_succeeded = float(scaler.get_scale()) >= scale_before" in source
    assert "step_succeeded=step_succeeded" in source
    assert source.index("scaler.update()") < source.index("csf_memory.update(")
    assert "SCENE_PROMPT_CSF_EPOCH" in source
    assert "SCENE_PROMPT_CSF_DONE" in source
    assert "declining_for_twenty_epochs" not in source
    assert "early_stop_on_decline" not in source
    assert "break" not in source
    assert "feature_dim=768" in source
    assert "trainable_parameters=0" in source

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    train = functions["do_train_stage2"]
    arguments = [arg.arg for arg in train.args.args]
    for required in [
        "csf_memory",
        "csf_weight",
        "csf_temperature",
        "csf_reconstruction_weight",
    ]:
        assert required in arguments

    print(
        "CSF_STAGE2_CONTRACT_OK",
        {
            "feature_branch": "feat[1]",
            "feature_dim": 768,
            "memory_update": "after_successful_optimizer_step",
            "xmodal_stacked": False,
            "early_stop": "disabled",
        },
    )


if __name__ == "__main__":
    main()
