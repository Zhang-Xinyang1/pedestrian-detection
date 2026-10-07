# CSF balanced_modality 30-epoch screening policy

This independent package implements the user-directed one-GPU screening gate.
Each method trains exactly 30 Stage 2 epochs and evaluates at epochs 10, 20, and 30.
A method passes only when at least two of the three checkpoints strictly exceed both UAD mAP and Rank-1.
A rejected method is not promoted to a 120-epoch run; the next method must use a new output directory.
This is an engineering filter, not a claim of a fair paper comparison because the historical UAD recipe differs from the CSF recipe.
\nThis package screens the balanced_modality prototype arm.\n