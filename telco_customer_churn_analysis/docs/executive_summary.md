# Executive Summary – Financially Optimized Churn Strategy

This report presents a complete, financially optimized customer retention strategy. The telecom provider faces a 26% churn rate with an average CLTV loss of $4,238 per churned customer — a material financial risk.

## Strategic Mandate: Maximize Profit
Using a Random Forest Classifier (Recall ≈ 88%) and SHAP explainability, we evaluated the profit impact of different probability thresholds.

The Profit Curve analysis concludes:

- **Optimal threshold:** 0.113
- **Maximum expected profit:** $1,253,729  
- **Mandate:** The 0.113 threshold must be used for all flagging, as it maximizes ROI.

## Key Risk Drivers (SHAP-Validated)
- **Fiber Optic Internet customers:** Highest churn risk  
- **New customers (0–12 months):** Early-life failure dominant  
- **Month-to-month contracts:** Structurally unstable  
- **Support quality issues:** Top qualitative churn driver  

## Deployment & Measurement
All flagged customers are routed through an A/B/C Testing Framework to scientifically measure the ROI of interventions before mass rollout.

This ensures the strategy is both financially sound and scientifically validated.