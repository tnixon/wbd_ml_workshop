# Databricks notebook source
# MAGIC %md
# MAGIC # Model validation
# MAGIC
# MAGIC When managing models in Unity Catalog (UC), we typically make use of different catalogs for each "environment" (ie. dev, test, prod).
# MAGIC In order to promote a model version to another environment, we simply copy the model version (including artifacts and metadata) from 
# MAGIC the lower environment catalog into the corresponding schema in the higher environment. 

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the source challenger model to promote

# COMMAND ----------

model_name = f"churn_model_{current_user_no_at}"
src_model_path = f"{dev_catalog}.{dbName}.{model_name}"
src_model_alias = "Challenger"

src_model_details = client.get_registered_model(src_model_path)
src_mv = client.get_model_version_by_alias(src_model_path, src_model_alias)

# COMMAND ----------

src_mv.description

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the current champion model (if any)

# COMMAND ----------

dest_model_path = f"{prod_catalog}.{dbName}.{model_name}"
prod_alias = "Champion"

try:
  cur_prod_mv = client.get_model_version_by_alias(dest_model_path, prod_alias)
except Exception:
  cur_prod_mv = None
  print(f"No current champion model exists!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy the model between catalogs and apply the Champion alias

# COMMAND ----------

# register a new model if it doesn't exist
try:
  client.get_registered_model(dest_model_path)
except Exception:
  client.create_registered_model(dest_model_path, 
                                 tags=src_model_details.tags, 
                                 description=src_model_details.description)
  
# copy the challenger model up to the new catalog
new_prod_mv = client.create_model_version(name=dest_model_path, 
                                          source=f"models:/{src_model_path}/{src_mv.version}",
                                          run_id=src_mv.run_id,
                                          tags=src_mv.tags, 
                                          description=src_mv.description)

# COMMAND ----------

# apply the champion alias to the new model version (in both catalogs)
client.set_registered_model_alias(name=dest_model_path, alias=prod_alias, version=new_prod_mv.version)
client.set_registered_model_alias(name=src_model_path, alias=prod_alias, version=src_mv.version)

# COMMAND ----------


