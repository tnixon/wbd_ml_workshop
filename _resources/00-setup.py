# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ./00-global-setup $reset_all_data=$reset_all_data $db_prefix=retail $min_dbr_version=12 $catalog="wbd_ml_workshop"

# COMMAND ----------

churn_schema = f"""
  customer_id string,
  gender string,
  senior_citizen int,
  partner string,
  dependents string,
  tenure int,
  phone_service string,
  multiple_lines string,
  internet_service string,
  online_security string,
  online_backup string,
  device_protection string,
  tech_support string,
  streaming_tv string,
  streaming_movies string,
  contract string,
  paperless_billing string,
  payment_method string,
  monthly_charges double,
  total_charges double,
  churn string
"""

# COMMAND ----------

import mlflow
if "evaluate" not in dir(mlflow):
    raise Exception("ERROR - YOU NEED MLFLOW 2.0 for this demo. Select DBRML 12+")
    
from databricks.feature_store import FeatureStoreClient
from mlflow import MlflowClient
from io import StringIO
import urllib

# define some UC paths
catalog = spark.catalog.currentCatalog()
db_name = spark.catalog.currentDatabase()
shared_schema = "default"
source_volume = "source"
source_path = f"/Volumes/{catalog}/{shared_schema}/{source_volume}"
shared_bronze = f"{catalog}.{shared_schema}.churn_bronze_customers"
db_bronze = f"{catalog}.{db_name}.churn_bronze_customers"

# should we reset the source data?
reset_all = dbutils.widgets.get("reset_all_data") == "true"
if reset_all:
  # delete the source data
  spark.sql(f"drop volume if exists {catalog}.{shared_schema}.{source_volume}")
  # delete the bronze tables
  spark.sql(f"drop table if exists {shared_bronze}")
  spark.sql(f"drop table if exists {db_bronze}")

# create the shared source volume if it doesn't exists
spark.sql(f"create schema if not exists {catalog}.{shared_schema}")
spark.sql(f"create volume if not exists {catalog}.{shared_schema}.{source_volume}")

# rebuild our source & bronze data
if reset_all or len(dbutils.fs.ls(source_path)) < 1:
  # download source data to our source volume
  # Dataset under apache license: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE
  src_data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
  dest_path = f"{source_path}/Telco-Customer-Churn.csv"
  urllib.request.urlretrieve(src_data_url, dest_path)

  # parse the CSV to a DataFrame
  churn_src_df = spark.read\
                  .format("csv")\
                  .option("header", "true")\
                  .schema(churn_schema)\
                  .load(f"dbfs:{dest_path}")
  
  # save the data to the bronze table
  churn_src_df.write\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .saveAsTable(shared_bronze)

# clone the bronze table into the individual schema
spark.sql(f"create table if not exists {db_bronze} shallow clone {shared_bronze}")

# COMMAND ----------

def display_automl_churn_link(table_name, force_refresh = False): 
  if force_refresh:
    reset_automl_run("churn_auto_ml")
  display_automl_link("churn_auto_ml", "dbdemos_mlops_churn", spark.table(table_name), "churn", 5)

def get_automl_churn_run(table_name = "dbdemos_mlops_churn_features", force_refresh = False):
  if force_refresh:
    reset_automl_run("churn_auto_ml")
  from_cache, r = get_automl_run_or_start("churn_auto_ml", "dbdemos_mlops_churn", spark.table(table_name), "churn", 5)
  return r

# COMMAND ----------

import re
from datetime import datetime
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# COMMAND ----------

# Replace this with your Slack webhook
slack_webhook = ""


# COMMAND ----------

# MAGIC %run ./API_Helpers
