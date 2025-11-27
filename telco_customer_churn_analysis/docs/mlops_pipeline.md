# MLOps Pipeline

## Components
- **Training pipeline**: ingestion → transform → model → evaluation
- **Deployment pipeline**: full retraining, serialization, S3 upload
- **Model artifact**: deployment_pipeline.pkl
- **Authentication**: Cognito
- **Model loading**: boto3 → S3

## Real-Time Inference
`predict_churn` function loads the model and applies thresholding.