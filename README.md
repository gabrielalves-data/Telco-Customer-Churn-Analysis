# 📊 Telco Customer Churn Analysis

![CI](https://github.com/gabrielalves-data/Telco-Customer-Churn-Analysis/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/Cloud-AWS-232F3E?logo=amazon-aws&logoColor=white)
![Render](https://img.shields.io/badge/Deployment-Render-46E3B7?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

* MIT License

## 🎥 App Demo

[![Watch Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/q_E4YNjwJQA)

## 📖 Overview
This project delivers an end-to-end, cloud-deployed Machine Learning system for predicting customer churn and optimizing retention campaign profitability. Unlike traditional notebook-only analyses, this repository provides:
* A high-recall Random Forest model (≈ 88%)
* SHAP explainability for interpretable decisions
* Profit Curve Optimization identifying the financial threshold that maximizes campaign ROI
* A secure cloud deployment (Render + AWS Cognito + S3)
* A production-ready web application for real-time inference
* A fully packaged Python library available on PyPI
* An A/B/C testing engine for scientific ROI measurement
This system transforms churn management into a measurable, financially optimized business process.

## 🧠 Key Highlight: Financial Optimization
A profit-driven analysis determined:
* Optimal intervention threshold: 0.113
* Projected campaign profit: $1,253,729
* Strategic mandate: All customer flagging must use the 0.113 threshold, as it maximizes expected ROI.
This approach ensures churn prevention is not just predictive — it’s profit-optimized.

## 🏗 Deployment Architecture
```
graph TD
    User([User]) -->|HTTPS / Custom Domain| CF[AWS CloudFront CDN]
    CF -->|Security Rules| WAF[AWS WAF]
    WAF -->|Traffic Forwarding| Render[Render Platform]
    
    subgraph Render Platform
        Docker[Docker Container]
        App[Python Application]
    end
    
    Render --> Docker
    Docker --> App
    
    subgraph AWS Cloud Services
        Cognito[AWS Cognito]
        S3[AWS S3 Bucket]
    end
    
    App -->|Auth / Login| Cognito
    App -->|Load Models & Assets| S3
```

## 📦 Installation
```pip install telco_customer_churn_analysis```

### Local Setup
```
git clone https://github.com/gabrielalves-data/Telco-Customer-Churn-Analysis.git
cd Telco-Customer-Churn-Analysis
pip install -r requirements.txt
```

### Environment Variables (.env)
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

COGNITO_USER_POOL_ID=...
COGNITO_APP_CLIENT_ID=...

S3_BUCKET_NAME=your_bucket
```

## 🧱 Project Structure
```
Telco-Customer-Churn-Analysis/
├── .github/ workflows/  # CI/CD configurations
├── telco_customer_churn_analysis/
    |── src/
        |── static/ eda/                    # Saved Images
        |── telco_customer_churn_analysis/
            |── cli.py                                  # Runnig using client
            |── model_utils.py                          # Functions for model training and tunning
            |── model_xai.py                            # Functions for model explanation
            |── s3_utils.py                             # S3 helper functions to save objects on AWS S3
            |── telco_customer_churn_analysis.py        # Base Functions
            |── telco_customer_churn_analysis_app.py    # Base Functions Adapted for Web App
            |── utils.py                                # General use util functions
        |── templates/                      # HTML and CSS Files
        ├── application.py                  # Main application entry point
    |── tests/           # PyTest Test Functions
    |── docs/            # Documentation   
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 📚 Documentation
All detailed sections — methodology, financial optimization, SHAP analysis, data dictionary, A/B/C testing, deployment — are available at:
* [Index](telco_customer_churn_analysis/docs/index.md)
* [Data Dictionary](telco_customer_churn_analysis/docs/data_dictionary.md)
* [Problem Statment](telco_customer_churn_analysis/docs/problem_statment.md)
* [Methodology](telco_customer_churn_analysis/docs/methodology.md)
* [Executive Summary](telco_customer_churn_analysis/docs/executive_summary.md)
* [MlOps Pipeline](telco_customer_churn_analysis/docs/mlops_pipeline.md)
* [SHAP Explainability](telco_customer_churn_analysis/docs/shap_explainability.md)
* [Profit Optimization](telco_customer_churn_analysis/docs/profit_optimization.md)
* [ABC Testing](telco_customer_churn_analysis/docs/abc_testing.md)

## 👨‍💻 Author
Gabriel Alves

## 📄 License
MIT License — see the `LICENSE` file.

## 🏆 Credits
This package was generated using:
* Cookiecutter
* audreyfeldroy/cookiecutter-pypackage template