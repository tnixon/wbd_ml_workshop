# Databricks notebook source
# MAGIC %run ../_resources/workshop_config

# COMMAND ----------

bronze_updates_df = spark.table(prod_user_update)

# COMMAND ----------

import re
import pyspark.pandas as ps

def compute_churn_features(data):
  
  # Convert to a dataframe compatible with the pandas API
  data = data.pandas_api()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phone_service', 'multiple_lines', 'internet_service',
                                 'online_security', 'online_backup', 'device_protection',
                                 'tech_support', 'streaming_tv', 'streaming_movies',
                                 'contract', 'paperless_billing', 'payment_method'], dtype = 'int64')
  
  # Convert label to int and rename column
  data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churn': 'int32'})
  
  # Clean up column names
  data.columns = [re.sub(r'[\(\)]', ' ', name).lower() for name in data.columns]
  data.columns = [re.sub(r'[ -]', '_', name).lower() for name in data.columns]

  
  # Drop missing values
  data = data.dropna()
  
  return data.to_spark()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# transform features
churn_features_df = compute_churn_features(bronze_updates_df)

# create the feature table if it doesn't exist
try:
  fe.get_table(name=prod_user_features)
except Exception:
  fe.create_table(
    name=prod_user_features,
    primary_keys=['customer_id'],
    schema=churn_features_df.schema,
    description='These features are derived from the churn_bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
  )

# COMMAND ----------

fe.write_table(df=churn_features_df, 
               name=prod_user_features,
               mode='merge')

# COMMAND ----------


