# Battery Health Prediction Using Machine Learning

This project predicts lithium-ion battery **State of Health (SOH)** from operational and electrochemical measurements. It evaluates several regression algorithms, ensemble approaches, and model-explainability techniques to identify an accurate and interpretable SOH prediction workflow.

## Overview

Battery health estimation is important for battery-management systems, electric vehicles, and energy-storage applications. The notebook trains and compares machine-learning models using battery-cycle data, then evaluates their predictive accuracy and explains influential features using SHAP.

The target variable is **SOH**. The input features are:

| Feature | Description |
| --- | --- |
| `Cycle` | Charge/discharge cycle number |
| `Voltage` | Battery voltage measurement |
| `Current` | Battery current measurement |
| `Temperature` | Battery temperature measurement |
| `Capacity` | Measured battery capacity |

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regression (RBF kernel)
- CatBoost Regressor
- Hybrid ensemble
- Stacking ensemble

The initial hybrid model averages Random Forest, XGBoost, and CatBoost predictions. The notebook also evaluates an optimized weighted hybrid model:

`0.25 × Random Forest + 0.60 × CatBoost + 0.15 × SVM`

## Methodology

1. Load the battery dataset and remove the non-numeric battery identifier when present.
2. Select `Cycle`, `Voltage`, `Current`, `Temperature`, and `Capacity` as predictors.
3. Apply Min-Max normalization to features and the SOH target.
4. Split the data into training (80%) and testing (20%) sets using `random_state=42`.
5. Train each regression model and evaluate it with MAE, MSE, RMSE, and R².
6. Create comparison, actual-versus-predicted, scatter, residual, feature-importance, and SHAP plots.

## Results

The notebook was run with 1,407 records: 1,125 training samples and 282 test samples. Metrics below are calculated on the Min-Max-normalized SOH target.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.16994 | 0.23049 | 0.22208 |
| Decision Tree | 0.00875 | 0.07501 | 0.91761 |
| Random Forest | 0.01338 | 0.06111 | 0.94532 |
| XGBoost | 0.01538 | 0.05656 | 0.95316 |
| SVM | 0.03749 | 0.08022 | 0.90577 |
| CatBoost | 0.02507 | 0.05606 | 0.95399 |
| Hybrid Ensemble | 0.01645 | **0.05450** | **0.95650** |
| Stacking Ensemble | 0.01721 | 0.05499 | 0.95572 |

Among the initially compared models, the hybrid ensemble achieved the strongest test-set performance, with an R² of **0.9565** and RMSE of **0.0545**.

## Explainability

The project uses two complementary methods to interpret model behavior:

- **Random Forest feature importance** for a global ranking of input influence.
- **SHAP (SHapley Additive exPlanations)** summary and bar plots for Random Forest, XGBoost, and CatBoost models.

These analyses help show how cycle count, voltage, current, temperature, and capacity contribute to SOH predictions.

## Project Outputs

When the notebook runs, it creates the following folders:

```text
Battery_Model_Plots/
├── Correlation_Heatmap.png
├── Overall_Model_Results.csv
├── Model_Comparison_R2.png
├── Feature_Importance.png
├── *_Actual_vs_Predicted.png
├── *_Scatter_Plot.png
└── *_Residual_Plot.png

Hybrid_Model_HD_Plots/
├── Hybrid_Actual_vs_Predicted_HD.png
├── Hybrid_Scatter_Plot_HD.png
└── Hybrid_Residual_Plot_HD.png

SHAP_HD_Plots/
├── SHAP_Summary_Plot_HD.png
└── SHAP_Bar_Plot_HD.png
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or Google Colab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap
```

### Dataset Setup

Place the dataset in the project directory with this expected name:

```text
NASA_Battery_SOC_SOH_RUL.csv
```

The CSV must include `SOH` and the five feature columns listed above. A `Battery` column is optional; the notebook drops it before modeling.

### Run the Project

1. Rename the supplied notebook file with an `.ipynb` extension if needed, for example `battery_health_prediction.ipynb`.
2. Open it in Jupyter Notebook or upload it to Google Colab.
3. Upload or place `NASA_Battery_SOC_SOH_RUL.csv` where the notebook can access it.
4. Run the cells in order.
5. Review the generated metrics, plots, and explainability outputs.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib and Seaborn
- SHAP
- Google Colab / Jupyter Notebook

## Notes and Limitations

- The reported metrics are based on a random train/test split. For time-dependent battery datasets, cycle-aware or battery-wise validation is recommended before deployment.
- Values are evaluated after target normalization; convert predictions back to the original SOH scale when reporting operational results.
- This repository is intended for research and educational use. A production battery-management system should be validated on representative field data and under relevant safety requirements.

## Future Improvements

- Add cross-validation grouped by battery identifier.
- Perform hyperparameter optimization for all models.
- Add Remaining Useful Life (RUL) and State of Charge (SOC) prediction pipelines.
- Save trained models and preprocessing scalers for inference.
- Build a dashboard or API for real-time SOH prediction.

## Author
KAMALNATH D S, 
M.KUMARASAMY COLLEGE OF ENGINEERING, 
Email : dskamalnath@gmail.com
