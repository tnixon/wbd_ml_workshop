# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Feature Engineering
# MAGIC Our first step is to analyze the data and build the features we'll use to train our model. Let's see how this can be done.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization Logic
# MAGIC
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# Read into Spark
telcoDF = spark.table("churn_bronze_customers")
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Pandas on spark
# MAGIC
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `Pandas on spark` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `pandas_api`.*

# COMMAND ----------

# DBTITLE 1,Define featurization function
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
  
  return data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Write to Feature Store (Optional)
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

churn_features_df = compute_churn_features(telcoDF)
tableName = 'dbdemos_mlops_churn_features'
try:
  #drop table if exists
  fe.drop_table(f'{dev_catalog}.{dbName}.{tableName}')
except:
  pass
#Note: You might need to delete the FS table using the UI
churn_feature_table = fe.create_table(
  name=f'{dev_catalog}.{dbName}.{tableName}',
  primary_keys=['customer_id'],
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the churn_bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fe.write_table(df=churn_features_df.to_spark(), name=f'{dev_catalog}.{dbName}.{tableName}')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient. 
# MAGIC
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC
# MAGIC
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`dbdemos_mlops_churn_features`)
# MAGIC
# MAGIC Our prediction target is the `churn` column.
# MAGIC
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

# DBTITLE 1,Start an AutoML run
from databricks import automl
from datetime import datetime

tableName = 'dbdemos_mlops_churn_features'
xp_path = f"/Users/{current_user}/databricks_automl/{dbName}"
xp_name = f"automl_{dbName}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

automl_run = automl.classify(
    experiment_name = xp_name,
    experiment_dir = xp_path,
    dataset = fe.read_table(name = f'{dev_catalog}.{dbName}.{tableName}'),
    target_col = "churn",
    exclude_frameworks = ["lightgbm"],
    timeout_minutes = 10
    ) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using the generated notebook to build our model
# MAGIC
# MAGIC Next step: [Explore the generated Auto-ML notebook]($./02_automl_baseline)
