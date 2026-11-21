# Inspection workload for industrial fault screening

How often can a team inspecting 5% of records recover at least 80% of faults? This study evaluates ten stratified record-level splits on Scania APS, AI4I and Conveyor, with logistic regression and random forests. Inspection means review of one flagged record.

## Inspect the results

| Dataset | Random forest median recall at 5% capacity | Splits reaching 80% |
|---|---:|---:|
| Scania APS | 95.0% | 10/10 |
| AI4I | 79.4% | 4/10 |
| Conveyor | 29.3% | 0/10 |

Read stability_summary.csv for median, range and target attainment by policy. split_results.csv contains the 300 split-level evaluations. per1000 is the number of flagged records per 1,000. R80 and R90 select a threshold from calibration faults at the specified recall target with a one-sided 95% order-statistic rule. Top5% inspects the highest-scoring floor(0.05N) test records. The ten splits overlap, so frequencies describe partition sensitivity.

## Reproduce the tables from the included predictions

From the repository root, use Python 3.12.14 and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python summarize.py
```

The verification checks threshold decisions against archived test scores. The summary script rebuilds both CSV tables. runs/42 through runs/51 contain scores, labels, zero-based row identifiers, thresholds and the workload curves. No sensor measurements are required for this route.

## Refit the models

Place the exact files described in [data/README.md](data/README.md) under data/. From the repository root run:

```bash
sha256sum -c checksums.txt
bash repeat.sh
```

This regenerates each seed directory, checks the decisions and rebuilds the summaries. Splits allocate 60% to training, 20% to threshold calibration and 20% to testing. The study evaluates retrospective screening at record level. Performance on new machines or later operating periods requires separate evaluation.
