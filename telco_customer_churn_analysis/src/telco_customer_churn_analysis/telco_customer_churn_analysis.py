import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")

filename = 'Telco_customer_churn.xlsx'

full_path = os.path.join(path, filename)

df = pd.read_excel(full_path)