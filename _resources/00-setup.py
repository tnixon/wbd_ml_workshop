# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
reset_all = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

# MAGIC %run ./workshop_config

# COMMAND ----------

import re

# VERIFY DATABRICKS VERSION COMPATIBILITY ----------

min_required_version = 12

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "

# COMMAND ----------

reserved_catalogs = ['hive_metastore', 'spark_catalog']

def create_and_use_catalog(catalog):
  # test if catalog exists already
  # you'll get a permissions error if you try to create one without the CREATE CATALOG permission
  # ---- even if you do 'create catalog if not exists'!
  all_catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
  # create it only if it doesn't exist yet! (assume we have permissions)
  if catalog not in all_catalogs and catalog not in reserved_catalogs:
    print(f"creating catalog {catalog}")
    spark.sql(f"create catalog if not exists {catalog}")
  # use it!
  print(f"using catalog {catalog}")
  spark.sql(f"USE CATALOG {catalog}")

def create_db(catalog, dbName):
  create_and_use_catalog(catalog)
  print(f"creating db {catalog}.{dbName}")
  spark.sql(f"create database if not exists `{dbName}`")

def drop_db(schema):
  print(f'dropping db {schema} and all objects in it')
  spark.sql(f"DROP DATABASE IF EXISTS `{schema}` CASCADE")

def drop_table(table):
  print(f'dropping table {table}')
  spark.sql(f"DROP TABLE IF EXISTS `{table}`")

def drop_volume(volume):
  print(f'dropping volume {volume}')
  spark.sql(f"DROP VOLUME IF EXISTS `{volume}`")

# COMMAND ----------

# drop tables & volumes if we're doing a reset
if reset_all:
  drop_db(dev_user_schema)
  drop_db(prod_user_schema)
  drop_table(shared_bronze)
  drop_table(shared_update)
  drop_volume(source_volume)

# COMMAND ----------

# create all our catalogs & schmas...
create_db(prod_catalog, dbName)
create_db(dev_catalog, dbName)
create_db(dev_catalog, shared_schema)

# create our source volume
print(f"creating source volume {source_volume}")
spark.sql(f"create volume if not exists {source_volume}")

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

from io import StringIO
import urllib

# rebuild our source & bronze data
if reset_all:
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
  
  # split off some records to use for an update
  bronze_df, update_df = churn_src_df.randomSplit([0.9, 0.1], seed=42)
  
  # save the data to the delta tables
  bronze_df.write\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .saveAsTable(shared_bronze)
  update_df.write\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .saveAsTable(shared_update)

# COMMAND ----------

# clone the shared tables into the individual schema
spark.sql(f"create table if not exists {dev_user_bronze} shallow clone {shared_bronze}")
spark.sql(f"create table if not exists {prod_user_bronze} shallow clone {shared_bronze}")
spark.sql(f"create table if not exists {prod_user_update} shallow clone {shared_update}")

# COMMAND ----------

spark.sql(f"USE CATALOG {dev_catalog}")
spark.sql(f"USE DATABASE {dbName}")

# COMMAND ----------


