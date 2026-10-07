# Single GPU serial screening of the remaining prototype arms

The user keeps identity as a backup and authorizes testing the remaining
six_scene and csf arms sequentially using exactly one GPU allocation.
Each arm starts from the approved Stage1 checkpoint, with reset memory,
seed1, P16K4, batch64, the existing augmentations, I2T=1.0, 30 Stage2 epochs,
and WHU all-same-camera exclusion. Loss weight=0.1, temperature=0.07,
reconstruction weight=0.1. Evaluate epochs10/20/30. At least two checkpoints
must strictly exceed both historical UAD metrics to pass the screening gate.
No arm is automatically extended beyond 30 epochs. Each arm has its own
output directory. Existing source, logs, checkpoints and outputs are preserved.
Training, gate and artifact-verification Python code matches the completed,
verified identity30 and balanced_modality30 packages. Both remaining methods
also passed the v3 five-arm A800 smoke previously. This release changes only
the serial submission script and package documentation. An inline read-only
check compares the actual starting state and protocol after both arms finish.
After submission report one initial status and a suggested user check time.
Do not poll, set an automation, or reserve another GPU.
