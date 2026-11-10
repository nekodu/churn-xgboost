# Telco churn: RandomForest vs XGBoost

End-to-end tabular machine learning on the IBM Telco churn dataset: data inspection and cleaning, feature engineering, model comparison with cross-validation, hyperparameter search and export of the final model for serving through [xgboost-fastapi](https://github.com/nekodu/xgboost-fastapi). Built in 2022.

**Result:** tuned XGBoost reaches **ROC-AUC 0.844** and F1 0.584 in 5-fold cross-validation, up from 0.822 / 0.553 for the untuned baseline.

## Skills shown

- **Data cleaning:** catching a numeric column stored as text, tracing the 11 resulting blanks to customers with zero tenure, and removing an ID column before it leaks into the features.
- **Exploratory analysis:** churn rate per category, numeric profiles per class, class balance (26.5% churners).
- **Feature engineering:** tenure and spend bands, service counts, contract engagement and service-combination flags.
- **Preprocessing:** automatic split into categorical, numeric and high-cardinality columns; label and one-hot encoding; outlier-robust scaling.
- **Modelling and evaluation:** RandomForest vs XGBoost, `GridSearchCV`, k-fold cross-validation on accuracy, F1 and ROC-AUC, validation curves.
- **Hand-off to production:** fitted models exported and served by a separate API.

## Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), included in `data/`. 7,043 customers and 21 columns. IDs are synthetic; the dataset contains no real personal data.

## Pipeline

1. **Inspect** shape, types, missing values and quantiles of every column.
2. **Fix types.** `TotalCharges` is text. Converting it exposes 11 blanks, all customers with tenure 0, which are dropped (7,032 rows remain).
3. **Type columns** with a helper that separates categorical, numeric and cardinal columns, so `customerID` is identified and dropped.
4. **Analyse the target** per category and per numeric column.
5. **Engineer features:** tenure bands, monthly and total spend bands, number of services, an "engaged" flag for one- and two-year contracts, service-combination flags.
6. **Encode and scale:** label encoding for binary columns, one-hot for the rest, `RobustScaler` for numeric columns.
7. **Model:** baseline comparison, grid search for both models, cross-validated final scores.
8. **Export** both fitted models for the API.

## Results (from the committed notebook outputs)

| Model | Setting | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| RandomForest | default, 10-fold CV | 0.790 | 0.553 | 0.826 |
| XGBoost | default, 5-fold CV | 0.784 | 0.553 | 0.822 |
| **XGBoost** | **tuned, 5-fold CV** | **0.803** | **0.584** | **0.844** |

Best XGBoost: `learning_rate=0.01, max_depth=5, n_estimators=500, colsample_bytree=0.5`.
Best RandomForest: `max_depth=8, max_features=5, min_samples_split=100, n_estimators=500`.

## What I'd improve

The notebook is kept as written in 2022. Looking at it now:

- **Make the evaluation strict.** Scaling is fitted before cross-validation and the grid search reuses the evaluation data, so the scores are somewhat optimistic. I'd wrap preprocessing in a scikit-learn `Pipeline`, keep a held-out test set and report nested cross-validation.
- **Optimise for the business metric.** With 26.5% churners, predicting "no churn" for everyone already gives 73.5% accuracy. I'd tune for F1 or recall and choose the decision threshold deliberately.
- **Small fixes.** The tuned RandomForest score is computed into `cv_resultsalpha` but the cell prints the earlier `cv_results`, so it is missing from the table (the exported model is the tuned one, since `set_params` works in place). `Contract_Extra_Services` duplicates `NEW_TotalServices`.
- **Safer model files.** Pickle executes code on load. I'd export with XGBoost's JSON format.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook xgboost.ipynb
```

Versions are pinned to 2022 releases (pandas 1.5, scikit-learn 1.1, xgboost 1.6).
