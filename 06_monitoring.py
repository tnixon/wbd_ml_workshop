# Databricks notebook source
# MAGIC %md
# MAGIC # Set up a Model Monitor
# MAGIC
# MAGIC We need to be able to track our model performance to watch for drift and check for bias.
# MAGIC [Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html) gives us the tools to do this easily from within the Databricks platform.

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.4-py3-none-any.whl"

# COMMAND ----------

# This step is necessary to reset the environment with our newly installed wheel.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/workshop_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Baseline table

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()

# configure model coordinates
prod_alias = "Champion"

# fetch the model version info
champ_mv = client.get_model_version_by_alias(prod_model_path, prod_alias)

# fetch the model
champ_model_uri = f"models:/{prod_model_path}@{prod_alias}"
champ_churn_model = mlflow.pyfunc.spark_udf(spark, champ_model_uri, result_type="int")

# COMMAND ----------

# load feature table
churn_features_df = spark.table(prod_user_features)

# split off baseline
baseline_df, predictions_df = churn_features_df.randomSplit([0.25, 0.75], seed=42)

# COMMAND ----------

import pyspark.sql.functions as sfn

def do_inference(features_df):
  model_features = champ_churn_model.metadata.get_input_schema().input_names()
  return features_df.withColumn("predition_time", sfn.current_timestamp())\
    .withColumn("churn_prediction", champ_churn_model(*model_features))\
    .withColumn("model_name", sfn.lit(prod_model_path))\
    .withColumn("model_alias", sfn.lit(prod_alias))\
    .withColumn("model_version", sfn.lit(champ_mv.version))

# COMMAND ----------

# run inference on baseline
do_inference(baseline_df).write.format("delta").mode("append").saveAsTable(prod_baseline)
# generate additional preditions
do_inference(predictions_df).write.format("delta").mode("append").saveAsTable(prod_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure monitoring

# COMMAND ----------

import databricks.lakehouse_monitoring as lm

# COMMAND ----------

print(f"Creating monitor for {prod_predictions}")

info = lm.create_monitor(
  table_name=prod_predictions,
  profile_type=lm.InferenceLog(
    timestamp_col="predition_time",
    granularities=["5 minutes"],
    model_id_col="model_version", # Model version number 
    prediction_col="churn_prediction",
    problem_type="classification",
    label_col="churn" # Optional
  ),
  baseline_table_name=prod_baseline,
  slicing_exprs=["gender_female=1", "senior_citizen=1"],
  output_schema_name=prod_user_schema
)

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status == lm.MonitorStatus.PENDING:
  print(f"Waiting for monitor {prod_predictions} to be created...")
  info = lm.get_monitor(table_name=prod_predictions)
  time.sleep(10)

assert info.status == lm.MonitorStatus.ACTIVE, "Error creating monitor"

# COMMAND ----------

lm.get_monitor(table_name=prod_predictions)

# COMMAND ----------

# A metric refresh will automatically be triggered on creation
refreshes = lm.list_refreshes(table_name=prod_predictions)
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (lm.RefreshState.PENDING, lm.RefreshState.RUNNING):
  print(f"Monitor refresh {run_info.refresh_id} is currently {run_info.state}")
  run_info = lm.get_refresh(table_name=prod_predictions, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert run_info.state == lm.RefreshState.SUCCESS, "Monitor refresh failed"

# COMMAND ----------


