# Notebook protocol

Run notebooks manually in numeric order. The repository audit reads saved JSON only and never executes a cell.

1. `01_Business_and_Data_Audit.ipynb` — scope, data license and leakage audit.
2. `02_Face_Detection_and_Quality.ipynb` — detection and input-quality behavior.
3. `03_Face_Verification_Experiments.ipynb` — model comparison and threshold lock on LFW fold 8.
4. `04_Liveness_Detection_Experiments.ipynb` — subject-disjoint synthetic PAD proxy model selection and checkpoint export.
5. `05_Business_Report.ipynb` — one-time LFW fold 9 and PAD holdout evaluation.

Notebook 05 has already consumed the current holdout once. Do not re-run its unlocked holdout cells merely to refresh output. A new evaluation requires a new experiment/model version and a documented holdout policy.
