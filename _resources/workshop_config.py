# Databricks notebook source
# catalogs
catalog_prefix = "wbd_mlworkshop"
dev_catalog = f"{catalog_prefix}_dev"
prod_catalog = f"{catalog_prefix}_prod"

# COMMAND ----------

# shared resources
shared_schema = "default"
source_volume_name = "source"
source_volume = f"{dev_catalog}.{shared_schema}.{source_volume_name}"
source_path = f"/Volumes/{dev_catalog}/{shared_schema}/{source_volume_name}"
shared_bronze = f"{dev_catalog}.{shared_schema}.churn_bronze_customers"
shared_update = f"{dev_catalog}.{shared_schema}.churn_updates"

# COMMAND ----------

import re

# user specific resources
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# user
db_prefix = "retail"
dbName = db_prefix+"_"+current_user_no_at

# COMMAND ----------

# user-level databases & tables
dev_user_schema = f"{dev_catalog}.{dbName}"
dev_user_bronze = f"{dev_catalog}.{dbName}.churn_bronze_customers"
prod_user_schema = f"{prod_catalog}.{dbName}"
prod_user_bronze = f"{prod_catalog}.{dbName}.churn_bronze_customers"
prod_user_update = f"{prod_catalog}.{dbName}.churn_updates"
