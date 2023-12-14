# Databricks notebook source
# MAGIC %run ../_resources/workshop_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Inference

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

# COMMAND ----------

import pyspark.sql.functions as sfn

model_features = champ_churn_model.metadata.get_input_schema().input_names()
churn_preditions = churn_features_df.withColumn("predition_time", sfn.current_timestamp())\
  .withColumn("churn_prediction", champ_churn_model(*model_features))\
  .withColumn("model_name", sfn.lit(prod_model_path))\
  .withColumn("model_alias", sfn.lit(prod_alias))\
  .withColumn("model_version", sfn.lit(champ_mv.version))

# COMMAND ----------

churn_preditions.write.format("delta").mode("append").saveAsTable(prod_predictions)

# COMMAND ----------


