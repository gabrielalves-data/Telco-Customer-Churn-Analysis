# Methodology & Approach

## 1. Discovery & EDA
- Demographics, service usage, billing behavior analyzed
- Chi² test confirmed strong correlation between contract type and churn
- Engineered features including Tenure Group

## 2. Modeling
- Multiple models evaluated
- Random Forest chosen for strongest Recall (~88%)
- Extensive cross-validation

## 3. Explainability
- SHAP used to validate global and local risk drivers
- Key factors: Fiber Optic, New Customer, Contract type, Charges

## 4. Financial Optimization
- Profit Curve used to convert predictions into financial value
- Optimal threshold = 0.113
- Generates projected profit = $1,253,729

## 5. Deployment
- Final model packaged as deployment_pipeline.pkl
- Loaded dynamically from AWS S3
- Real-time scoring exposed via application