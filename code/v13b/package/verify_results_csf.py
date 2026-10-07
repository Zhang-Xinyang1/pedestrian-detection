"""Formal CPU verifier for CSF Stage 2-only artifacts."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from prototype_comparison import METHODS, PrototypeComparisonMemory
from starting_point_regularization import COEFFICIENT, PREFIX, compare_visual_states


PACKAGE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON value")


def verify_matrix(matrix, size, name):
    require(len(matrix) == size, f"Wrong {name} matrix row count")
    require(all(len(row) == size for row in matrix), f"Wrong {name} matrix column count")
    for row in matrix:
        for cell in row:
            if cell is None:
                continue
            require(set(cell) == {"ap", "r1", "valid_queries", "skipped_queries"}, f"Wrong {name} matrix cell schema")
            finite(cell)
            require(cell["valid_queries"] > 0, f"Empty valid cell in {name} matrix")


def load_memory(directory, audit, summary):
    memory_path = directory / "csf_memory.pt"
    memory_audit_path = directory / "csf_memory_audit.json"
    require(memory_path.is_file(), "Missing CSF memory state")
    require(memory_audit_path.is_file(), "Missing CSF memory audit")
    memory_record = read(memory_audit_path)
    require(sha256(memory_path) == audit["memory_file_sha256"] == memory_record["file_sha256"], "Memory file SHA256 differs")

    payload = torch.load(memory_path, map_location="cpu", weights_only=False)
    require(payload["num_classes"] == 500 and payload["feature_dim"] == 768, "Wrong memory dimensions")
    memory = PrototypeComparisonMemory(
        payload["num_classes"],
        payload["feature_dim"],
        momentum=float(payload["momentum"]),
        ridge=float(payload["ridge"]),
    )
    memory.load_state_dict(payload["state_dict"], strict=True)
    state_sha = memory.state_sha256()
    require(state_sha == payload["state_sha256"], "Serialized memory state hash differs")
    require(state_sha == memory_record["state_sha256"], "Memory audit state hash differs")
    require(state_sha == audit["memory_state_sha256"], "Stage 2 memory state hash differs")
    require(state_sha == summary["memory_state_sha256"], "CSF summary memory state hash differs")
    require(sum(parameter.numel() for parameter in memory.parameters()) == 0, "Memory contains trainable parameters")

    state = memory.state_dict()
    require(set(state) == set(memory_record["tensors"]), "Memory tensor schema differs")
    for name, tensor in state.items():
        record = memory_record["tensors"][name]
        require(list(tensor.shape) == record["shape"], "Memory tensor shape differs: " + name)
        require(str(tensor.dtype) == record["dtype"], "Memory tensor dtype differs: " + name)
        require(tensor_sha256(tensor) == record["sha256"], "Memory tensor hash differs: " + name)
        if name == "factor_condition":
            valid = memory.factor_valid
            require(not bool(valid.any()) or bool(torch.isfinite(tensor[valid]).all()), "Valid factor condition is nonfinite")
        elif tensor.is_floating_point():
            require(bool(torch.isfinite(tensor).all()), "Nonfinite memory tensor: " + name)

    require(int(memory.successful_updates.item()) == audit["optimizer_steps"], "Memory update count differs from successful optimizer steps")
    require(int(memory.skipped_updates.item()) == audit["planned_steps"] - audit["optimizer_steps"], "Memory skipped-step count differs")
    require(int(memory.identity_valid.sum().item()) == memory_record["initialized_identities"], "Initialized identity count differs")
    require(int(memory.factor_valid.sum().item()) == memory_record["factor_valid_identities"], "Factor-valid identity count differs")
    require(int(memory.factor_valid.sum().item()) > 0, "No factor-valid identities")
    return memory, memory_record


def evaluation_summary(result):
    scenes = {int(key): float(value) for key, value in result["scenes"].items()}
    ground = sum(scenes[index] for index in range(3)) / 3.0
    aerial = sum(scenes[index] for index in range(3, 6)) / 3.0
    modalities = {
        str(modality): (scenes[modality] + scenes[modality + 3]) / 2.0
        for modality in range(3)
    }
    directions = []
    for source, row in enumerate(result["matrices"]["scene"]):
        for target, cell in enumerate(row):
            if cell is not None:
                directions.append((float(cell["ap"]), source, target))
    require(directions, "No valid scene-matrix direction")
    worst_direction = min(directions)
    worst_scene = min((value, scene) for scene, value in scenes.items())
    return {
        "map_percent": 100.0 * result["ap"],
        "rank1_percent": 100.0 * result["r1"],
        "rank5_percent": 100.0 * result["r5"],
        "rank10_percent": 100.0 * result["r10"],
        "macro_map_percent": 100.0 * result["macro_map"],
        "worst_map_percent": 100.0 * result["worst_map"],
        "ground_macro_map_percent": 100.0 * ground,
        "aerial_macro_map_percent": 100.0 * aerial,
        "modality_macro_map_percent": {key: 100.0 * value for key, value in modalities.items()},
        "worst_scene": {"scene": worst_scene[1], "map_percent": 100.0 * worst_scene[0]},
        "worst_direction": {
            "source_scene": worst_direction[1],
            "target_scene": worst_direction[2],
            "map_percent": 100.0 * worst_direction[0],
        },
    }


def validate_epoch_budget(audit, smoke=False, screening_30=False, screening_40=False, fixed100_ab=False,smoke_epochs=4,fixed80_cooldown=False):
    require(sum((smoke,screening_30,screening_40,fixed100_ab,fixed80_cooldown)) <= 1, "Ambiguous epoch budget flags")
    require(smoke_epochs in (2,4), 'Unsupported smoke budget')
    expected = smoke_epochs if smoke else (80 if fixed80_cooldown else (40 if screening_40 else (30 if screening_30 else 100)))
    require(audit["final_epoch"] == expected, "Incomplete fixed epoch budget")
    require(audit["csf"]["final_epoch"] == expected, "CSF summary final epoch differs")
    require(not audit["stopped_early"] and not audit["csf"]["stopped_early"], "Metric early stopping is disabled")
    require(audit["early_stop_reason"] is None and audit["csf"]["early_stop_reason"] is None, "Unexpected early-stop reason")
    return expected


def verify(directory, expected_seed, smoke=False, require_targets=False, screening_30=False, screening_40=False, fixed100_ab=False,smoke_epochs=4,fixed80_cooldown=False):
    require(directory.is_dir(), "Output directory does not exist")
    artifact_manifest_path = directory / "ARTIFACT_SHA256.json"
    marker_path = directory / "VERIFICATION.json"
    require(not artifact_manifest_path.exists() and not marker_path.exists(), "Verification outputs already exist")

    required_base = [
        "manifest.json",
        "stage1_checkpoint_load.json",
        "stage2_initial_parameters.json",
        "stage2_audit.json",
        "csf_memory.pt",
        "csf_memory_audit.json",
    ]
    for name in required_base:
        require((directory / name).is_file(), "Missing artifact: " + name)

    manifest = read(directory / "manifest.json")
    load_record = read(directory / "stage1_checkpoint_load.json")
    initial = read(directory / "stage2_initial_parameters.json")
    audit = read(directory / "stage2_audit.json")
    summary = audit["csf"]
    cfg = manifest["config"]
    info = manifest["info"]
    finite(manifest)
    finite(audit)

    require(expected_seed in (1, 2, 3), "Expected seed must be 1, 2, or 3")
    require(info["seed"] == expected_seed == cfg["SOLVER"]["SEED"], "Seed contract differs")
    require(info["data_protocol"] == "unchanged_official_P16K4", "Data protocol differs")
    require(info["train_metadata_sha256"] == manifest["provenance"]["train_metadata_sha256"], "Training metadata hash differs")
    require(info["feature_dim"] == 1280 and info["csf_feature_dim"] == 768, "Feature dimensions differ")
    require(cfg["MODEL"]["I2T_LOSS_WEIGHT"] == 1.0, "I2T loss weight differs")
    require(cfg["DATALOADER"]["NUM_INSTANCE"] == (2 if smoke else 4), "Instances-per-identity differs")
    require(cfg["SOLVER"]["STAGE2"]["IMS_PER_BATCH"] == (4 if smoke else 64), "Stage 2 batch size differs")
    expected_epochs = smoke_epochs if smoke else (80 if fixed80_cooldown else (40 if screening_40 else (30 if screening_30 else 100)))
    if smoke:
        require(info.get('smoke_epochs')==smoke_epochs,'Declared smoke budget differs')
    require(cfg["SOLVER"]["STAGE2"]["MAX_EPOCHS"] == expected_epochs, "Stage 2 epoch count differs")
    require(bool(info.get("screening_30", False)) == bool(screening_30), "Screening budget flag differs")
    require(bool(info.get("screening_40", False)) == bool(screening_40), "Screening-40 budget flag differs")
    require(bool(info.get("fixed100_ab", False)) == bool(fixed100_ab), "Fixed120 cooldown flag differs")
    require(bool(info.get("fixed80_cooldown", False)) == bool(fixed80_cooldown), "Fixed80 cooldown flag differs")
    if fixed100_ab or fixed80_cooldown:
        require(cfg["SOLVER"]["STAGE2"]["STEPS"] == [21], "Fixed120 learning-rate milestones differ")
        require(cfg["SOLVER"]["STAGE2"]["BASE_LR"] == 5e-6 and cfg["SOLVER"]["STAGE2"]["GAMMA"] == 0.1, "Fixed120 learning-rate values differ")
        require(cfg["SOLVER"]["STAGE2"]["CHECKPOINT_PERIOD"] == 10, "Fixed120 checkpoint period differs")
    require(cfg["SOLVER"]["STAGE2"]["EVAL_PERIOD"] == (smoke_epochs if smoke else 10), "Evaluation period differs")
    require(not cfg["MODEL"]["SIE_CAMERA"] and not cfg["MODEL"]["SIE_VIEW"], "SIE protocol differs")
    require(cfg["TEST"]["FEAT_NORM"] == "yes" and not cfg["TEST"]["RE_RANKING"], "Retrieval protocol differs")

    expected_sources = {
        "run_stage2_only.py",
        "scene_stage2_csf.py",
        "factorized_prototype.py",
        "prototype_comparison.py",
        "run_official.py",
        "whu_adapter.py",
        "whu_metrics.py",
        "scene_prompt.py",
        "quality_gate.py",
        "starting_point_regularization.py",
        "l2sp_diagnostics.py",
        "local_observed_auxiliary.py",
        "verify_local_auxiliary.py",
        "partial_local_matching.py",
        "lr_tail_schedule.py",
        "training_state_io.py",
    }
    require(set(manifest["adapter_sources"]) == expected_sources, "Source provenance set differs")
    for name, digest in manifest["adapter_sources"].items():
        require(sha256(PACKAGE / name) == digest, "Source hash differs: " + name)
    for name, digest in manifest["provenance"]["runtime_upstream_hashes"].items():
        require(sha256(PACKAGE / "upstream" / name) == digest, "Upstream hash differs: " + name)

    source_checkpoint = Path(load_record["source_path"])
    require(source_checkpoint.is_file(), "Source Stage 1 checkpoint is unavailable")
    source_sha = sha256(source_checkpoint)
    approved = read(PACKAGE / "APPROVED_STAGE1.json")
    require(source_sha == approved["file_sha256"], "Source checkpoint differs from approved SHA256")
    require(initial == approved["parameter_hashes"], "Initial parameters differ from approved Stage 1 audit")
    require(load_record["loaded_tensor_hash"] == approved["tensor_hash"], "Loaded state differs from approved Stage 1 state")
    require(source_sha == load_record["source_file_sha256"], "Source Stage 1 checkpoint SHA256 differs")
    require(source_sha == info["source_stage1_checkpoint_sha256"], "Manifest Stage 1 checkpoint SHA256 differs")
    require(source_sha == audit["source_stage1_checkpoint_sha256"], "Stage 2 audit source SHA256 differs")
    require(load_record["source_tensor_hash"] == load_record["loaded_tensor_hash"], "Loaded Stage 1 tensors differ")
    require(not load_record["missing_keys"] and not load_record["unexpected_keys"], "Stage 1 strict-load contract failed")

    final_epoch = validate_epoch_budget(audit, smoke, screening_30, screening_40, fixed100_ab,smoke_epochs,fixed80_cooldown)
    require(manifest["info"]["early_stop_policy"]["enabled"] is False, "Early-stop policy must be disabled")
    require(summary["early_stop_policy"]["enabled"] is False, "Training enabled early stopping")
    final_checkpoint = directory / audit["final_checkpoint"]
    require(final_checkpoint.name == f"ViT-B-16_{final_epoch}.pth", "Wrong final checkpoint name")
    require(final_checkpoint.is_file(), "Missing final checkpoint")
    checkpoint_sha = sha256(final_checkpoint)
    require(checkpoint_sha == audit["final_checkpoint_sha256"], "Final checkpoint whole-file SHA256 differs")
    state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    require(state["classifier.weight"].shape == (500, 768), "Classifier shape differs")
    require(state["classifier_proj.weight"].shape == (500, 512), "Projected classifier shape differs")
    require(state["prompt_learner.cls_ctx"].shape == (500, 4, 512), "Prompt shape differs")
    for name, tensor in state.items():
        require(bool(torch.isfinite(tensor).all()), "Nonfinite checkpoint tensor: " + name)

    require(set(initial) == set(audit["final_parameters"]), "Parameter schema changed")
    for name, digest in audit["final_parameters"].items():
        require(name in state, "Checkpoint missing audited parameter: " + name)
        require(tensor_sha256(state[name]) == digest, "Checkpoint parameter fingerprint differs: " + name)
    changed = [name for name in initial if initial[name] != audit["final_parameters"][name]]
    require(sorted(changed) == sorted(audit["changed"]), "Changed-parameter list differs")
    require(set(changed).issubset(set(audit["optimizer_parameters"])), "Changed parameter was absent from optimizer")
    require(
        sorted(audit["optimizer_parameters"]) == load_record["optimizer_parameters"],
        "Loaded optimizer range differs from final audit",
    )
    require(audit["frozen_parameters"] == load_record["frozen_parameters"], "Frozen range differs")
    require(not set(changed).intersection(audit["frozen_parameters"]), "A frozen parameter changed")
    require(any(name.startswith("image_encoder.") for name in changed), "Visual encoder did not update")
    require("classifier.weight" in changed and "classifier_proj.weight" in changed, "Classifiers did not update")
    require(not any(name.startswith(("text_encoder.", "prompt_learner.")) for name in changed), "Frozen text or prompt parameter changed")
    require(0 < audit["optimizer_steps"] <= audit["planned_steps"], "Optimizer-step count differs")
    require(len(audit["actual_batches_per_epoch"]) == final_epoch, "Incomplete epoch accounting")
    require(sum(audit["actual_batches_per_epoch"]) == audit["planned_steps"], "Batch budget differs")

    summary = audit["csf"]
    require(summary["feature_branch"] == "identity_768" and summary["feature_dim"] == 768, "Wrong CSF feature branch")
    require(summary["trainable_parameters"] == 0, "CSF added trainable parameters")
    require(summary["data_and_sampler_unchanged"], "CSF reports changed data or sampler")
    require(summary["weight"] == info["csf_loss_weight"], "CSF weight differs")
    require(summary["prototype_method"] == info["prototype_method"], "Prototype method differs")
    require(info["prototype_method"] in METHODS, "Unknown prototype method")
    require(summary["candidate_policy"] == info["candidate_policy"], "Candidate policy differs")
    require(summary["candidate_policy"] in ("matched_observed_scene_full_rank", "full_identity_with_factor_fallback", "full_observed_identity_center"), "Unknown candidate policy")
    require(summary["batches"] == audit["planned_steps"], "Prototype diagnostic batch count differs")
    if info["prototype_method"] == "control":
        require(summary["weight"] == 0.0 and summary["mean_loss"] == 0.0, "Control used prototype supervision")
    else:
        require(summary["weight"] == 0.1, "Fixed comparison weight differs")

    require(summary["temperature"] == info["csf_temperature"], "CSF temperature differs")
    require(summary["reconstruction_weight"] == info["csf_reconstruction_weight"], "CSF reconstruction weight differs")
    require(len(summary["epoch_history"]) == final_epoch, "Incomplete CSF epoch history")
    expected_evaluations = final_epoch // cfg["SOLVER"]["STAGE2"]["EVAL_PERIOD"]
    require(len(summary["evaluation_history"]) == expected_evaluations, "Incomplete evaluation history")
    if summary["weight"] > 0:
        require(summary["batches"] == audit["planned_steps"], "CSF batch count differs")
        require(summary["mean_loss"] > 0, "CSF loss was inactive")
        require(summary["mean_valid_anchor_fraction"] > 0, "CSF had no valid anchors")
        require(summary["mean_factor_target_fraction"] > 0, "CSF had no factor-valid targets")
        require(summary["successful_optimizer_steps"] == audit["optimizer_steps"], "CSF successful-step count differs")
        require(summary["skipped_optimizer_steps"] == audit["planned_steps"] - audit["optimizer_steps"], "CSF skipped-step count differs")

    if info.get('l2sp_enabled'):
        require(info['l2sp_coefficient']==COEFFICIENT,'L2SP coefficient differs')
        init_anchor=read(directory/'l2sp_initialization.json')
        anchor_audit=read(directory/'l2sp_audit.json')
        spec=anchor_audit['specification']
        require(spec==init_anchor['specification'],'L2SP specification changed')
        require(spec['coefficient']==COEFFICIENT and spec['trainable_parameters']==0,'L2SP configuration differs')
        names=sorted(n for n in initial if n.startswith(PREFIX))
        require(names==spec['parameter_names'],'Incorrect anchored parameter set')
        require(set(names).issubset(audit['optimizer_parameters']),'Anchored parameter absent from optimizer')
        require(init_anchor==manifest['l2sp'],'L2SP manifest differs')
        require(init_anchor['initial_penalty']==0.0,'L2SP did not start at zero')
        require(init_anchor['source_stage1_sha256']==source_sha,'Anchor source differs')
        refs={n:initial[n] for n in names}
        require(refs==anchor_audit['reference_parameter_hashes']==anchor_audit['final_anchor_hashes']==init_anchor['reference_parameter_hashes'], 'Anchor reference changed')
        source_state=torch.load(source_checkpoint,map_location='cpu',weights_only=False)
        drift=compare_visual_states(source_state,state)
        for key,value in drift.items():
            require(math.isclose(value,anchor_audit['final_drift'][key],rel_tol=1e-5,abs_tol=1e-9),'Parameter drift differs: '+key)
        require(drift['weighted_penalty']>0.0,'L2SP never affected updated visual parameters')
        require(spec['parameter_elements']==sum(source_state[n].numel() for n in names),'Anchor size differs')
        history=anchor_audit['epoch_history']
        require(history==summary['l2sp_epoch_history'] and len(history)==final_epoch,'L2SP accounting differs')
        for index,row in enumerate(history,1):
            finite(row)
            require(row['epoch']==index and row['batches']==audit['actual_batches_per_epoch'][index-1],'L2SP batch budget differs')
        probes=read(directory/'training_diagnostics.json')
        require({row['epoch'] for row in probes}==set(range(1,final_epoch+1)),'Missing training diagnostics epochs')
        require(any(row['gradient_norms']['l2sp']>0 for row in probes),'No regularization gradient recorded')
        for row in probes:finite(row)
        cp_period=cfg['SOLVER']['STAGE2']['CHECKPOINT_PERIOD']
        require(cp_period==(smoke_epochs if smoke else 10),'Diagnostic checkpoint period differs')
        for epoch in range(cp_period,final_epoch+1,cp_period):
            cp_record=read(directory/f'checkpoint_audit_stage2_{epoch:03d}.json')
            cp_path=directory/f'ViT-B-16_{epoch}.pth'
            require(cp_record['epoch']==epoch and cp_record['file']==cp_path.name,'Intermediate checkpoint schema differs')
            require(sha256(cp_path)==cp_record['sha256'],'Intermediate checkpoint file changed')
            require(set(cp_record['parameters'])==set(initial),'Checkpoint audit parameter schema differs')
            require(all(cp_record['parameters'][n]==initial[n] for n in audit['frozen_parameters']),'Intermediate frozen parameter changed')
            branches=read(directory/f'branch_diagnostics_stage2_{epoch:03d}.json')
            finite(branches)
            require(branches['epoch']==epoch and branches['diagnostic_only'] and branches['official_descriptor_dim']==1280,'Branch diagnostic scope differs')
            require(branches['protocol']=='WHU_all_same_camera_excluded','Branch diagnostic protocol differs')
            require(set(branches['branch_metrics'])=={'identity768','projection512','concat1280'},'Missing branch metrics')
            require(all(r['gallery']==(24 if smoke else 93609) for r in branches['branch_metrics'].values()),'Branch gallery differs')
        del source_state
    memory, memory_record = load_memory(directory, audit, summary)
    local_verification = None
    if info.get('local_observed_enabled'):
        from verify_local_auxiliary import verify_local_auxiliary
        local_verification = verify_local_auxiliary(directory,audit,manifest,memory,smoke=smoke)

    period = cfg["SOLVER"]["STAGE2"]["EVAL_PERIOD"]
    evaluations = []
    for epoch in range(period, final_epoch + 1, period):
        path = directory / f"eval_stage2_{epoch:03d}.json"
        require(path.is_file(), "Missing evaluation artifact: " + path.name)
        result = read(path)
        finite(result)
        require(result["epoch"] == epoch, "Evaluation epoch differs")
        require(result["local_smoke"] == smoke, "Evaluation smoke mode differs")
        require(result["feature_dim"] == 1280, "Retrieval feature dimension differs")
        require(result["protocol"] == "WHU_all_same_camera_excluded", "Evaluation protocol differs")
        require(result["valid_queries"] + result["skipped_queries"] == result["queries"], "Query accounting differs")
        if not smoke:
            require(result["queries"] == 6405 and result["gallery"] == 93609, "Production evaluation scope differs")
            require(result["valid_queries"] == 6405 and result["skipped_queries"] == 0, "Production valid-query scope differs")
        require(set(result["scenes"]) == {str(index) for index in range(6)}, "Missing six-scene metrics")
        require(abs(result["macro_map"] - sum(result["scenes"].values()) / 6.0) < 1e-12, "Macro scene mAP differs")
        require(abs(result["worst_map"] - min(result["scenes"].values())) < 1e-12, "Worst scene mAP differs")
        verify_matrix(result["matrices"]["scene"], 6, "scene")
        verify_matrix(result["matrices"]["modality"], 3, "modality")
        evaluations.append(result)

    final_result = evaluations[-1]
    from lr_tail_schedule import verify_tail_history
    from training_state_io import verify_training_state
    tail_verified=verify_tail_history(directory,manifest,audit)
    training_state_verified=verify_training_state(directory,manifest,audit)

    if require_targets:
        require(final_result["ap"] >= 0.12, "Formal mAP target not reached")
        require(final_result["r1"] >= 0.30, "Formal Rank-1 target not reached")

    final_summary = evaluation_summary(final_result)
    final_summary["lr_tail_control"] = tail_verified
    final_summary["training_state"] = training_state_verified
    if local_verification is not None:
        final_summary['local_auxiliary'] = local_verification
    final_summary.update(
        {
            "seed": expected_seed,
            "prototype_method": info["prototype_method"],
            "checkpoint": final_checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "memory_state_sha256": memory.state_sha256(),
            "factor_valid_identities": memory_record["factor_valid_identities"],
            "initialized_identities": memory_record["initialized_identities"],
            "targets_required": bool(require_targets),
            "targets_passed": final_result["ap"] >= 0.12 and final_result["r1"] >= 0.30,
            "screening_30": bool(screening_30),
            "screening_40": bool(screening_40),
            "fixed100_ab": bool(fixed100_ab),
            "fixed80_cooldown": bool(fixed80_cooldown),
            "final_epoch": final_epoch,
            "stopped_early": bool(summary["stopped_early"]),
            "early_stop_reason": summary["early_stop_reason"],
        }
    )
    summary_path = directory / "CSF_FINAL_SUMMARY.json"
    summary_path.write_text(json.dumps(final_summary, indent=2, sort_keys=True), encoding="utf-8")

    artifact_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in {"ARTIFACT_SHA256.json", "VERIFICATION.json"}
    )
    artifact_hashes = {
        str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
        for path in artifact_files
    }
    artifact_manifest_path.write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    artifact_manifest_sha = sha256(artifact_manifest_path)
    marker = {
        "status": "PASS",
        "lr_tail_control": tail_verified,
        "training_state": training_state_verified,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": expected_seed,
        "smoke": bool(smoke),
        "targets_required": bool(require_targets),
        "targets_passed": final_summary["targets_passed"],
        "screening_30": bool(screening_30),
        "screening_40": bool(screening_40),
            "fixed100_ab": bool(fixed100_ab),
            "fixed80_cooldown": bool(fixed80_cooldown),
        "l2sp_enabled": bool(info.get('l2sp_enabled')),
        "l2sp_coefficient": info.get('l2sp_coefficient',0.0),
        "map_percent": final_summary["map_percent"],
        "rank1_percent": final_summary["rank1_percent"],
        "final_epoch": final_epoch,
        "stopped_early": bool(summary["stopped_early"]),
        "early_stop_reason": summary["early_stop_reason"],
        "checkpoint": final_checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "memory_state_sha256": memory.state_sha256(),
        "artifact_manifest": artifact_manifest_path.name,
        "artifact_manifest_sha256": artifact_manifest_sha,
        "artifact_count": len(artifact_hashes),
    }
    if local_verification is not None:
        marker['local_auxiliary'] = local_verification
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    print("CSF_ARTIFACTS_OK " + json.dumps(marker, sort_keys=True))
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-seed", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--smoke-epochs",type=int,choices=[2,4],default=4)
    parser.add_argument("--require-targets", action="store_true")
    parser.add_argument("--screening-30", action="store_true")
    parser.add_argument("--screening-40", action="store_true")
    parser.add_argument("--fixed80-cooldown", action="store_true")
    parser.add_argument("--fixed100-ab", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    verify(
        arguments.directory,
        expected_seed=arguments.expected_seed,
        smoke=arguments.local_smoke,
        require_targets=arguments.require_targets,
        screening_30=arguments.screening_30,
        screening_40=arguments.screening_40,
        fixed100_ab=arguments.fixed100_ab,
        fixed80_cooldown=arguments.fixed80_cooldown,
        smoke_epochs=arguments.smoke_epochs,
    )
