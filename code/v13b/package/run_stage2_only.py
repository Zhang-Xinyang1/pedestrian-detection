"""Run CSF Stage 2 from an explicitly approved Stage 1 checkpoint."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

import run_official as base
import scene_stage2_csf as stage2
from prototype_comparison import METHODS, PrototypeComparisonMemory
from starting_point_regularization import StartingPointRegularizer, COEFFICIENT
from l2sp_diagnostics import diagnostic_evaluator_class
from local_observed_auxiliary import initialize_auxiliary, state_hashes, mapping_sha
from lr_tail_schedule import MatchedTailScheduler, tail_specification


PACKAGE = Path(__file__).resolve().parent


def tensor_hashes(state):
    result = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        result[name] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return result


def hash_mapping(mapping):
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def memory_audit(memory, path):
    state = memory.state_dict()
    tensors = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        if name == "factor_condition":
            valid = memory.factor_valid.detach().cpu()
            finite = bool(torch.isfinite(value[valid]).all()) if bool(valid.any()) else True
        else:
            finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": finite,
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
    payload = {
        "state_sha256": memory.state_sha256(),
        "tensors": tensors,
        "factor_valid_identities": int(memory.factor_valid.sum().item()),
        "initialized_identities": int(memory.identity_valid.sum().item()),
        "successful_updates": int(memory.successful_updates.item()),
        "skipped_updates": int(memory.skipped_updates.item()),
        "factor_metrics": memory.factor_metrics(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--stage1-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--prompt-mode", default="modality", choices=["modality"])
    parser.add_argument("--i2t-weight", type=float, default=1.0, choices=[1.0])
    parser.add_argument("--csf-weight", type=float, default=0.1, choices=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--csf-temperature", type=float, default=0.07)
    parser.add_argument("--csf-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--l2sp", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--smoke-epochs",type=int,choices=[2,4],default=4,
                        help="Explicit smoke-only budget; production remains fixed40")
    parser.add_argument("--screening-30", action="store_true",
                        help="Run the fixed 30-epoch screening budget")
    parser.add_argument("--screening-40", action="store_true",
                        help="Run the fixed 40-epoch cooldown screening budget")
    parser.add_argument("--fixed80-cooldown", action="store_true", help="Fixed80 V10; same first40 and LR5e-7 epoch21to80")
    parser.add_argument("--fixed100-ab", action="store_true",
                        help="Fixed100 AB; same V10 first40, one LR-tail variable after epoch40")
    parser.add_argument("--smoke-six-scenes", action="store_true")
    parser.add_argument("--prototype-method", choices=METHODS, default="csf")
    parser.add_argument("--local-observed", action="store_true",
                        help="V9 training-only observed foreign-scene local supervision")
    parser.add_argument("--local-transfer-disabled", action="store_true",
                        help="Explicit later ablation; never automatically submitted")
    parser.add_argument("--lr-tail-arm", choices=["A","B"], default="A")
    args = parser.parse_args()
    if not args.local_smoke and args.smoke_epochs!=4:
        raise ValueError('smoke-epochs applies only to a local-smoke run')
    if args.local_transfer_disabled and not args.local_observed:
        raise ValueError('Transfer ablation requires local supervision')
    if args.local_observed and (args.prototype_method != 'identity_full' or not args.l2sp
                               or not (args.local_smoke or args.screening_40 or args.fixed80_cooldown or args.fixed100_ab)):
        raise ValueError('V9 requires the matched identity_full + L2SP fixed40/smoke recipe')
    if args.l2sp and args.prototype_method not in ('csf','csf_full_identity','identity_full'):
        raise ValueError('L2SP experiment requires an approved prototype arm')
    if sum((args.screening_30,args.screening_40,args.fixed100_ab,args.fixed80_cooldown)) > 1:
        raise ValueError('Choose only one fixed budget')
    if args.prototype_method == "control":
        args.csf_weight = 0.0
    elif args.csf_weight <= 0:
        raise ValueError("Prototype arms require a positive weight")

    if args.output.exists():
        raise ValueError("Output exists; never overwrite")
    if not args.stage1_checkpoint.is_file():
        raise ValueError("Stage 1 checkpoint does not exist")
    if args.smoke_six_scenes and not args.local_smoke:
        raise ValueError("--smoke-six-scenes requires --local-smoke")
    if args.csf_temperature <= 0.0 or args.csf_reconstruction_weight < 0.0:
        raise ValueError("Invalid CSF configuration")

    provenance = json.loads((PACKAGE / "PROVENANCE.json").read_text(encoding="utf-8"))
    for name, digest in provenance["runtime_upstream_hashes"].items():
        if base.sha(PACKAGE / "upstream" / name) != digest:
            raise ValueError("Upstream source differs: " + name)

    cfg = base.config(args)
    cfg.defrost()
    if args.local_smoke:
        cfg.SOLVER.STAGE2.MAX_EPOCHS=args.smoke_epochs
        cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD=args.smoke_epochs
        cfg.SOLVER.STAGE2.EVAL_PERIOD=args.smoke_epochs
    if args.screening_30 or args.screening_40 or args.fixed100_ab or args.fixed80_cooldown:
        if args.local_smoke:
            raise ValueError("screening budget cannot be combined with --local-smoke")
        cfg.SOLVER.STAGE2.MAX_EPOCHS = 80 if args.fixed80_cooldown else (100 if args.fixed100_ab else (40 if args.screening_40 else 30))
        cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
        cfg.SOLVER.STAGE2.EVAL_PERIOD = 10
        cfg.SOLVER.STAGE2.STEPS = [21] if (args.screening_40 or args.fixed100_ab or args.fixed80_cooldown) else [60,100]
    cfg.SOLVER.SEED = args.seed
    cfg.freeze()

    seed_everything(args.seed)
    dataset = base.WHUMARS(args.data_root)
    if dataset.num_train_pids != 500:
        raise ValueError("Expected 500 training identities")
    if dataset.train_signature != provenance["train_metadata_sha256"]:
        raise ValueError("Training metadata hash differs")

    sampler_batch = 4 if args.local_smoke else 64
    sampler_instances = 2 if args.local_smoke else 4
    sampler = base.RandomIdentitySampler(dataset.train, sampler_batch, sampler_instances)
    planned_indices = list(iter(sampler))
    if len(planned_indices) % sampler_batch:
        raise ValueError("Sampler plan has an incomplete batch")
    if not args.local_smoke:
        for start in range(0, len(planned_indices), sampler_batch):
            labels = [dataset.train[index][1] for index in planned_indices[start : start + sampler_batch]]
            if len(set(labels)) != 16 or any(labels.count(label) != 4 for label in set(labels)):
                raise ValueError("P16K4 sampler contract differs")

    source_checkpoint_sha256 = base.sha(args.stage1_checkpoint)
    approved = json.loads((PACKAGE / "APPROVED_STAGE1.json").read_text(encoding="utf-8"))
    if source_checkpoint_sha256 != approved["file_sha256"]:
        raise ValueError("Stage 1 checkpoint differs from the approved file SHA256")
    info = {
        "classes": 500,
        "train": 92133,
        "query": 6405,
        "gallery": 93609,
        "seed": args.seed,
        "local_smoke": args.local_smoke,
        "stage2_epochs": cfg.SOLVER.STAGE2.MAX_EPOCHS,
        "screening_30": bool(args.screening_30),
        "screening_40": bool(args.screening_40),
        "fixed100_ab": bool(args.fixed100_ab),
        "lr_tail_arm": args.lr_tail_arm,
        "lr_tail_specification": tail_specification(args.lr_tail_arm, smoke=args.local_smoke),
        "fixed80_cooldown": bool(args.fixed80_cooldown),
        "l2sp_enabled": bool(args.l2sp),
        "l2sp_coefficient": COEFFICIENT if args.l2sp else 0.0,
        "checkpoint_selection": 'fixed_final_epoch',
        "branch_diagnostics_scope": 'fixed_up_to32_queries_per_scene_full_gallery',
        "stage2_plan_batches": len(planned_indices) // sampler_batch,
        "sampler_plan_sha256": hashlib.sha256(
            json.dumps(planned_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "feature_dim": 1280,
        "csf_feature_dim": 768,
        "prompt_mode": args.prompt_mode,
        "i2t_loss_weight": float(args.i2t_weight),
        "csf_loss_weight": float(args.csf_weight),
        "prototype_method": args.prototype_method,
        "candidate_policy": ("full_observed_identity_center" if args.prototype_method == "identity_full" else
                             ("full_identity_with_factor_fallback" if args.prototype_method == "csf_full_identity"
                              else "matched_observed_scene_full_rank")),
        "csf_temperature": float(args.csf_temperature),
        "csf_reconstruction_weight": float(args.csf_reconstruction_weight),
        "early_stop_policy": {
            "enabled": False,
            "evaluation_period": 10,
            "rule": "complete configured epoch budget",
        },
        "data_protocol": "unchanged_official_P16K4",
        "train_metadata_sha256": dataset.train_signature,
        "source_stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "source_stage1_checkpoint_sha256": source_checkpoint_sha256,
        "upstream_commit": provenance["commit"],
        "local_observed_enabled": bool(args.local_observed),
        "local_transfer_disabled": bool(args.local_transfer_disabled),
        "smoke_epochs": args.smoke_epochs if args.local_smoke else None,
    }
    print("CSF_STAGE2_INPUTS_OK " + json.dumps(info, sort_keys=True), flush=True)
    if args.check_only:
        print("CSF_STAGE2_PREFLIGHT_OK (no output writes or training)", flush=True)
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CSF Stage 2 requires exactly one visible CUDA GPU")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)

    if args.local_smoke:
        dataset = base.small_dataset(dataset, args.smoke_six_scenes)
    dataset.train = [
        (path, identity, camera, 3 * int(camera >= 5) + modality)
        for path, identity, camera, modality in dataset.train_meta
    ]

    manifest = {
        "info": info,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": base.yaml.safe_load(cfg.dump()),
        "provenance": provenance,
        "adapter_sources": {
            name: base.sha(PACKAGE / name)
            for name in [
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
            ]
        },
        "actual_run_counts": {
            "train": len(dataset.train),
            "query": len(dataset.query),
            "gallery": len(dataset.gallery),
        },
        "train_scene_counts": {
            str(scene): sum(row[3] == scene for row in dataset.train)
            for scene in range(6)
        },
        "eval_query": [
            str(Path(row[0]).relative_to(args.data_root)) for row in dataset.query_meta
        ],
        "eval_gallery": [
            str(Path(row[0]).relative_to(args.data_root)) for row in dataset.gallery_meta
        ],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    base.setup_logger("transreid", str(args.output), if_train=True)
    base.loaders.__factory["whumars"] = lambda root: dataset
    train2, _, validation, num_query, num_classes, num_cams, num_views = base.loaders.make_dataloader(cfg)
    train2 = base.CountingLoader(train2)

    def offline_download(url, *unused_args, **unused_kwargs):
        if url != base.models.clip._MODELS["ViT-B-16"]:
            raise ValueError("Unexpected weight request")
        return str(args.weights)

    base.models.clip._download = offline_download
    model = base.ScenePromptModel(num_classes, num_cams, num_views, cfg, quality_gate=False)
    base.attach_scene_prompts(model, args.weights, args.prompt_mode)
    manifest["prompt_spec"] = model.prompt_learner.specification()

    source_state = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source_state, dict) or not source_state:
        raise ValueError("Stage 1 checkpoint is not a state dictionary")
    incompatible = model.load_state_dict(source_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Stage 1 checkpoint load was not strict")
    if model.num_classes != 500 or model.in_planes != 768 or model.in_planes_proj != 512:
        raise ValueError("Model feature contract differs")

    loaded_state = model.state_dict()
    source_tensor_hashes = tensor_hashes(source_state)
    loaded_tensor_hashes = tensor_hashes(loaded_state)
    if source_tensor_hashes != loaded_tensor_hashes:
        raise ValueError("Loaded model tensors differ from Stage 1 checkpoint")
    if hash_mapping(loaded_tensor_hashes) != approved["tensor_hash"]:
        raise ValueError("Loaded model differs from the approved Stage 1 tensor hash")
    load_record = {
        "source_path": str(args.stage1_checkpoint.resolve()),
        "source_file_sha256": source_checkpoint_sha256,
        "source_tensor_hash": hash_mapping(source_tensor_hashes),
        "loaded_tensor_hash": hash_mapping(loaded_tensor_hashes),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "tensor_count": len(loaded_tensor_hashes),
    }
    (args.output / "stage1_checkpoint_load.json").write_text(
        json.dumps(load_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["stage1_checkpoint_load"] = load_record
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    loss_fn, center = base.make_loss(cfg, num_classes=num_classes)
    initial_parameters = base.fingerprints(model)
    if initial_parameters != approved["parameter_hashes"]:
        raise ValueError("Initial Stage 2 parameters differ from the verified Stage 1 audit")
    (args.output / "stage2_initial_parameters.json").write_text(
        json.dumps(initial_parameters, indent=2, sort_keys=True), encoding="utf-8"
    )

    optimizer, optimizer_center = base.make_optimizer_2stage(cfg, model, center)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_parameters = [
        names[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if any(name.startswith(("prompt_learner.", "text_encoder.")) for name in optimizer_parameters):
        raise ValueError("Stage 2 optimizer includes frozen text or prompt parameters")
    frozen_parameters = sorted(
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )
    load_record["optimizer_parameters"] = sorted(optimizer_parameters)
    load_record["frozen_parameters"] = frozen_parameters
    (args.output / "stage1_checkpoint_load.json").write_text(
        json.dumps(load_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["stage1_checkpoint_load"] = load_record
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    anchor = StartingPointRegularizer(model) if args.l2sp else None
    if anchor is not None:
        if not set(anchor.names).issubset(optimizer_parameters):
            raise ValueError('Visual anchor range omitted from optimizer')
        if anchor.initial_hashes != {n:initial_parameters[n] for n in anchor.names}:
            raise ValueError('Anchor differs from approved starting parameters')
        if anchor.anchor_hashes() != anchor.initial_hashes or float(anchor(model).detach()) != 0.0:
            raise ValueError('Starting-point penalty is nonzero at initialization')
        init_record = dict(specification=anchor.specification(),reference_parameter_hashes=anchor.initial_hashes,
            source_stage1_sha256=source_checkpoint_sha256,initial_penalty=0.0)
        (args.output/'l2sp_initialization.json').write_text(json.dumps(init_record,indent=2,sort_keys=True),encoding='utf-8')
        manifest['l2sp']=init_record
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
        print('L2SP_INITIALIZATION_OK ' + json.dumps(anchor.specification()),flush=True)
    optimizer_steps = [0]
    local_auxiliary = None
    if args.local_observed:
        local_auxiliary = initialize_auxiliary(model, args.seed, args.local_transfer_disabled)
        auxiliary_initial = {n: base.sha_tensor(p) if hasattr(base, 'sha_tensor') else
                             hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                             for n, p in local_auxiliary.named_parameters()}
        auxiliary_optimizer_names = []
        for name, parameter in local_auxiliary.named_parameters():
            auxiliary_optimizer_names.append(name)
            is_bias = 'bias' in name
            optimizer.add_param_group(dict(params=[parameter],
                lr=cfg.SOLVER.STAGE2.BASE_LR * (cfg.SOLVER.STAGE2.BIAS_LR_FACTOR if is_bias else 1),
                weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS if is_bias else cfg.SOLVER.STAGE2.WEIGHT_DECAY))
        assert not {id(p) for p in model.parameters()}.intersection(id(p) for p in local_auxiliary.parameters())
        auxiliary_init = dict(specification=local_auxiliary.specification(),
            auxiliary_seed=60000 + args.seed, main_rng_preserved=True,
            initial_parameters=auxiliary_initial, optimizer_parameters=auxiliary_optimizer_names,
            initial_state_sha256=mapping_sha(state_hashes(local_auxiliary)),
            initial_memory_observations=int(local_auxiliary.memory.mass.sum()),
            regularized_by_stage1_anchor=False, main_parameter_schema_unchanged=True)
        (args.output/'local_auxiliary_initialization.json').write_text(
            json.dumps(auxiliary_init,indent=2,sort_keys=True),encoding='utf-8')
        manifest['local_auxiliary']=auxiliary_init
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
        local_auxiliary.attach(model.image_encoder)
        print('LOCAL_OBSERVED_INITIALIZATION_OK '+json.dumps(auxiliary_init['specification'],sort_keys=True),flush=True)
    hook = optimizer.register_step_post_hook(
        lambda *_: optimizer_steps.__setitem__(0, optimizer_steps[0] + 1)
    )
    scheduler = MatchedTailScheduler(
        optimizer,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
        arm=args.lr_tail_arm, tail_start=2 if args.local_smoke else 40,
        tail_end=4 if args.local_smoke else 100,
    )
    stage2.R1_mAP_eval = base.evaluator_class(
        dataset,
        args.output,
        cfg.SOLVER.STAGE2.EVAL_PERIOD,
        cfg.SOLVER.STAGE2.MAX_EPOCHS,
        args.local_smoke,
    )

    stage2.R1_mAP_eval = diagnostic_evaluator_class(stage2.R1_mAP_eval,dataset,args.output,cfg.SOLVER.STAGE2.EVAL_PERIOD)
    memory = PrototypeComparisonMemory(500, 768, momentum=0.2, ridge=1e-4).cuda()
    print("CSF_STAGE2_START", flush=True)
    csf_summary, memory = stage2.do_train_stage2(
        cfg,
        model,
        center,
        train2,
        validation,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        0,
        csf_memory=memory,
        prototype_method=args.prototype_method,
        starting_point_regularizer=anchor,
        csf_weight=args.csf_weight,
        csf_temperature=args.csf_temperature,
        csf_reconstruction_weight=args.csf_reconstruction_weight,
        local_auxiliary=local_auxiliary,
    )
    hook.remove()
    auxiliary_audit = None
    if local_auxiliary is not None:
        local_auxiliary.detach()
        auxiliary_audit = local_auxiliary.summary()
        auxiliary_audit.update(initial_parameters=auxiliary_initial,
            final_parameters={n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                              for n,p in local_auxiliary.named_parameters()},
            optimizer_parameters=auxiliary_optimizer_names,
            joint_optimizer_parameter_count=len(optimizer_parameters)+len(auxiliary_optimizer_names))
        auxiliary_file = args.output/'local_auxiliary_final.pt'
        torch.save(dict(specification=local_auxiliary.specification(),state_dict=local_auxiliary.state_dict(),
                        state_sha256=auxiliary_audit['final_state_sha256']),auxiliary_file)
        auxiliary_audit['file_sha256']=base.sha(auxiliary_file)
        (args.output/'local_auxiliary_audit.json').write_text(
            json.dumps(auxiliary_audit,indent=2,sort_keys=True),encoding='utf-8')

    if anchor is not None:
        if anchor.anchor_hashes()!=anchor.initial_hashes:
            raise ValueError('Frozen regularizer reference mutated')
        anchor_audit=dict(specification=anchor.specification(),reference_parameter_hashes=anchor.initial_hashes,
            final_anchor_hashes=anchor.anchor_hashes(),final_drift=anchor.drift(model),
            epoch_history=csf_summary['l2sp_epoch_history'],reference_parameter_count=anchor.element_count)
        (args.output/'l2sp_audit.json').write_text(json.dumps(anchor_audit,indent=2,sort_keys=True),encoding='utf-8')
    final_parameters = base.fingerprints(model)
    changed = [
        name
        for name in initial_parameters
        if initial_parameters[name] != final_parameters[name]
    ]
    if optimizer_steps[0] <= 0 or not any(name.startswith("image_encoder.") for name in changed):
        raise ValueError("Stage 2 did not update the visual encoder")
    if any(name.startswith(("text_encoder.", "prompt_learner.")) for name in changed):
        raise ValueError("Stage 2 changed fixed text or prompt parameters")
    final_epoch = int(csf_summary["final_epoch"])
    if len(train2.counts) != final_epoch:
        raise ValueError("Stage 2 epoch accounting differs from the actual final epoch")
    if csf_summary["stopped_early"] or final_epoch != cfg.SOLVER.STAGE2.MAX_EPOCHS:
        raise ValueError("Incomplete Stage 2 epoch budget; metric early stopping is disabled")
    final_checkpoint = args.output / f"{cfg.MODEL.NAME}_{final_epoch}.pth"
    if not final_checkpoint.is_file():
        raise ValueError("Final Stage 2 checkpoint is missing")
    final_checkpoint_sha256 = base.sha(final_checkpoint)

    memory_file = args.output / "csf_memory.pt"
    torch.save(
        {
            "state_dict": memory.state_dict(),
            "num_classes": memory.num_classes,
            "feature_dim": memory.feature_dim,
            "momentum": memory.momentum,
            "ridge": memory.ridge,
            "state_sha256": memory.state_sha256(),
        },
        memory_file,
    )
    memory_record = memory_audit(memory, args.output / "csf_memory_audit.json")
    memory_record["file_sha256"] = base.sha(memory_file)
    (args.output / "csf_memory_audit.json").write_text(
        json.dumps(memory_record, indent=2, sort_keys=True), encoding="utf-8"
    )

    stage2_record = {
        "changed": changed,
        "optimizer_parameters": optimizer_parameters,
        "frozen_parameters": frozen_parameters,
        "optimizer_steps": optimizer_steps[0],
        "actual_batches_per_epoch": train2.counts,
        "planned_steps": sum(train2.counts),
        "source_stage1_checkpoint_sha256": source_checkpoint_sha256,
        "final_checkpoint": final_checkpoint.name,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "final_epoch": final_epoch,
        "stopped_early": bool(csf_summary["stopped_early"]),
        "early_stop_reason": csf_summary["early_stop_reason"],
        "csf": csf_summary,
        "memory_state_sha256": memory.state_sha256(),
        "memory_file_sha256": memory_record["file_sha256"],
        "final_parameters": final_parameters,
        "local_auxiliary": auxiliary_audit,
    }
    (args.output / "stage2_audit.json").write_text(
        json.dumps(stage2_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "CSF_STAGE2_DONE "
        + json.dumps(
            {
                key: value
                for key, value in stage2_record.items()
                if key not in {"changed", "optimizer_parameters", "final_parameters"}
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
