# Welcome to the Telco Customer Churn Analysis Documentation
Welcome to the full documentation for the **Telco Customer Churn Analysis** system — a fully production-ready, financially optimized machine learning framework designed to predict customer churn, optimize retention campaign ROI, and support real-time inference in a secure cloud environment.

This documentation provides everything you need to understand, use, extend, and deploy the system, from high-level business strategy to deep technical details.

# 📘 About the Project
This solution goes beyond traditional churn modeling by combining:

- **High-recall prediction (≈ 88%)** using a Random Forest Classifier  
- **Explainable AI (SHAP)** to identify and justify key churn drivers  
- **Profit Curve Optimization** to determine the **financially optimal threshold** (0.113)  
- **Cloud-native deployment** using AWS Cognito, S3, CloudFront, WAF, and Render  
- **A/B/C testing framework** to scientifically measure ROI from retention offers  
- **A published PyPI package** for enterprise-ready usage  
- **A Dockerized inference application** for consistent local and cloud execution  

The result is a complete, end-to-end, revenue-driven customer lifecycle management system.

# 🧠 Key Features
- **📈 Financially optimized churn prediction**  
  → Maximizes campaign profit ($1,253,729   expected return)
- **🔍 SHAP interpretability**  
  → Identifies the true drivers behind churn, such as service quality and tenure risks
- **⚙️ Full MLOps workflow**  
  → Model training, packaging, deployment, monitoring
- **☁️ Production cloud architecture**  
  → Secure login, S3 model storage, CDN acceleration, WAF protection
- **🧪 Built-in A/B/C testing engine**  
  → Validates campaign uplift before scaling interventions

---

# 🏗️ High-Level Architecture
The system includes:
* **Data layer**: S3-stored model artifacts
* **Authentication**: AWS Cognito
* **Inference Web App**: Docker + Render
* **Security**: AWS WAF + CloudFront
* **MLOps Pipelines**: Training, deployment, and evaluation
* **Experimentation**: A/B/C randomized uplift testing