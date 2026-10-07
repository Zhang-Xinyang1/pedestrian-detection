"""Static checks for the formal CSF verifier."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "verify_results_csf.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    required = [
        "--expected-seed",
        "--require-targets",
        "Formal mAP target not reached",
        "Formal Rank-1 target not reached",
        "Final checkpoint whole-file SHA256 differs",
        "Memory file SHA256 differs",
        "Memory update count differs from successful optimizer steps",
        "feature_dim"]
    for token in required:
        assert token in source, token
    assert "0.12" in source
    assert "0.30" in source
    assert "ARTIFACT_SHA256.json" in source
    assert "VERIFICATION.json" in source
    assert "CSF_FINAL_SUMMARY.json" in source
    assert "factor_valid_identities" in source
    assert "worst_direction" in source
    assert "ground_macro_map_percent" in source
    assert "aerial_macro_map_percent" in source
    assert "choices=[1, 2, 3]" in source
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"verify", "load_memory", "evaluation_summary", "verify_matrix"}.issubset(functions)
    print(
        "CSF_VERIFY_CONTRACT_OK",
        {
            "whole_checkpoint_sha256": True,
            "memory_state_audit": True,
            "same_checkpoint_targets": [12.0, 30.0],
            "supported_seeds": [1, 2, 3],
            "durable_marker": "VERIFICATION.json",
        },
    )


if __name__ == "__main__":
    main()
