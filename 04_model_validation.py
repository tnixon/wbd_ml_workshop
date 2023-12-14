# Databricks notebook source
# MAGIC %md
# MAGIC # Model validation
# MAGIC
# MAGIC This notebook execution can be automatically triggered from CI/CD to test the validity of new `challenger` models.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## General Validation Checks
# MAGIC
# MAGIC In the context of MLOps, there are more tests than simply how accurate a model will be.  To ensure the stability of our ML system and compliance with any regulatory requirements, we will subject each model added to the registry to a series of validation checks.  These include, but are not limited to:
# MAGIC <br><br>
# MAGIC * __Inference on production data__
# MAGIC * __Input schema ("signature") compatibility with current model version__
# MAGIC * __Accuracy on multiple slices of the training data__
# MAGIC * __Model documentation__
# MAGIC
# MAGIC In this notebook we explore some approaches to performing these tests, and how we can add metadata to our models with tagging if they have passed a given test or not.
# MAGIC
# MAGIC This part is typically specific to your line of business and quality requirement.
# MAGIC
# MAGIC For each test, we'll add information using tags to know what has been validated in the model. We can also add Comments if needed.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Model information
# MAGIC
# MAGIC Remember how webhooks can send data from one webservice to another?  With MLflow webhooks we send data about a model, and in the following cell we fetch that data to know which model is meant to be tested. 
# MAGIC
# MAGIC This is be done getting the `event_message` received by MLFlow webhook: `dbutils.widgets.get('event_message')`
# MAGIC
# MAGIC To keep things simple we use a helper function `fetch_webhook_data`, the details of which are found in the _API_Helpers_ notebook.  

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

model_name = f"churn_model_{current_user_no_at}"
model_path = f"{dev_catalog}.{dbName}.{model_name}"
model_alias = "Challenger"

# Get the challenger model in transition, its name and version from the metadata
client = mlflow.tracking.MlflowClient()

model_details = client.get_model_version_by_alias(model_path, model_alias)
run_info = client.get_run(run_id=model_details.run_id)

# COMMAND ----------

model_details

# COMMAND ----------

run_info

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Validate prediction
# MAGIC
# MAGIC We want to test to see that the model can predict on production data.  So, we will load the model and the latest from the feature store and test making some predictions.

# COMMAND ----------

############change to feature engineering 
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Read from feature store 
data_source = run_info.data.tags['db_table']
features = fe.read_table(name=data_source)

# Load model as a Spark UDF
model_path = f"{dev_catalog}.{dbName}.{model_name}"
model_uri = f'models:/{model_path}@{model_alias}'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Select the feature table cols by model input schema
input_column_names = loaded_model.metadata.get_input_schema().input_names()

# Predict on a Spark DataFrame
try:
  display(features.withColumn('predictions', loaded_model(*input_column_names)))
  client.set_model_version_tag(name=model_path, version=model_details.version, key="predicts", value=1)
except Exception: 
  print("Unable to predict on features.")
  client.set_model_version_tag(name=model_path, version=model_details.version, key="predicts", value=0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Demographic accuracy
# MAGIC
# MAGIC How does the model perform across various slices of the customer base?

# COMMAND ----------

import numpy as np
features_pd = features.withColumn('predictions', loaded_model(*input_column_names)).toPandas()
features_pd['accurate'] = np.where(features_pd.churn == features_pd.predictions, 1, 0)

# Check run tags for demographic columns and accuracy in each segment
try:
  demographics = run_info.data.tags['demographic_vars'].split(",")
  slices = features_pd.groupby(demographics).accurate.agg(acc = 'sum', obs = lambda x:len(x), pct_acc = lambda x:sum(x)/len(x))
  
  # Threshold for passing on demographics is 55%
  demo_test = "pass" if slices['pct_acc'].any() > 0.55 else "fail"
  
  # Set tags in registry
  client.set_model_version_tag(name=model_path, version=model_details.version, key="demo_test", value=demo_test)

  print(slices)
except KeyError:
  print("KeyError: No demographics_vars tagged with this model version.")
  client.set_model_version_tag(name=model_path, version=model_details.version, key="demo_test", value="none")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description check
# MAGIC
# MAGIC Has the data scientist provided a description of the model being submitted?

# COMMAND ----------

# If there's no description or an insufficient number of charaters, tag accordingly
if not model_details.description:
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_description", value=0)
  print("Did you forget to add a description?")
elif not len(model_details.description) > 20:
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_description", value=0)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_description", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Has the data scientist logged supplemental artifacts along with the original model?

# COMMAND ----------

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=model_path, version=model_details.version, key="has_artifacts", value=1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

results = client.get_model_version(name=model_path, version=model_details.version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulation, our model is now automatically tested and will be transitioned accordingly 
# MAGIC
# MAGIC We now have the certainty that our model is ready to be used as it matches our quality standard.
# MAGIC
# MAGIC
# MAGIC Next: [Run batch inference from our STAGING model]($./06_staging_inference)
