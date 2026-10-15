# prior-mismatch-tabpfn

Code and data for the paper **"Prior Mismatch in Tabular Foundation Models and How to Fix It"** (Artem Kultyshev, 2026).

TabPFN is overconfident on financial time series and loses its skill at market regime breaks, and three generations of the model (TabPFN-2, -3 and -3.5) share this failure. The paper traces both problems to the synthetic prior the models were trained on, and repairs both by fine-tuning on a realistic market prior: simulated stocks with volatility clustering, regime breaks and market-level noise, fitted to 20 calibration stocks kept apart from the test set.

## What is in this repository

| Folder | What it does | Paper |
|---|---|---|
| `Data/` | Daily open, high, low and close prices: 20 test stocks, 20 calibration stocks, 7 currencies | Section 3.4, Appendix A |
| `Step1_v35_analysis/` | TabPFN-2, TabPFN-3, TabPFN-3.5 and logistic regression on the test stocks and currencies. TabPFN-2 and TabPFN-3 predictions are read from `previous_versions_results/` | Section 4, Table 2 |
| `Step2_mechanism_explanation/` | Noise cross-test: small prior-fitted networks trained on a low-noise and a market-level-noise prior, each tested on both | Sections 5–6, Table 3 |
| `Step3_prior_calibration/` | The realistic market prior and its four-stage fit to the calibration stocks. `step3_prior_calibration_rules.txt` lists every rule of the prior | Section 7, Table 4, Appendix C |
| `Step4_tabpfn_v2_finetune/` | TabPFN-2 fine-tuned on the prior, three seeds, tested on both test sets | Section 8, Tables 5–6, Appendix D |
| `Step5_tabpfn_v35_finetune/` | The same for TabPFN-3.5 | Section 8, Tables 5–6, Appendix D |

Run the steps in order: Steps 4 and 5 read the fitted prior from `Step3_prior_calibration/results/pricepath_params.json`, which Step 3 writes.

## Environments

Two Python environments are needed, because TabPFN-2 and TabPFN-3.5 come from different versions of the `tabpfn` package.

- **tabpfn 2.0.8** with PyTorch (CUDA): Step 4.
- **tabpfn 9.x** with PyTorch (CUDA 12.8): Steps 1 and 5.
- Steps 2 and 3 run in either (Step 2 needs PyTorch; Step 3 needs only numpy, pandas and scikit-learn).

All steps also use pandas, numpy, scipy, scikit-learn and openpyxl; Steps 4 and 5 use `schedulefree`.

**Token.** If the TabPFN-3.5 weights need a token to download, set it as the environment variable `TABPFN_TOKEN` (on Windows: `setx TABPFN_TOKEN "your-token"`, then open a new console). Never write the token into a script.

## How to run

Each script has a `MODE` setting at the top:

- `"check"`: one stock and one currency, folder checks and a time estimate;
- `"run"`: everything (resumes from saved results where possible);
- `"analyse"`: tables only, from saved results.

Each script finds the `Data/` folder and the earlier steps' folders from the repository root, and writes its tables to a results text file in its own folder.

**Hardware.** One consumer GPU is enough. On an RTX 5070, the whole fine-tune experiment (three seeds and the evaluation on both test sets) takes about 1 hour for TabPFN-2 and about 2 hours for TabPFN-3.5, with 6.6 GB of peak GPU memory for TabPFN-3.5.

## Notes

- **Onset-shock setting in Step 3.** The code also searches an "onset shock" setting (a one-day volatility jump when a break begins). The fit set its size to 0, so it has no effect on the prior; the paper therefore leaves it out of Appendix C.
- **Script comments.** Some comments inside the scripts use older table numbers and the older name "price-path prior" for what the paper calls the realistic market prior.

## Citation

```
@article{kultyshev2026prior,
  title  = {Prior Mismatch in Tabular Foundation Models and How to Fix It},
  author = {Kultyshev, Artem},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
