# Databricks notebook source
# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC
# MAGIC By managing your [models in Unity Catalog](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html), we can addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Models in UC
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with UC.  Think of this as **committing** the model to UC, much as you would commit code to a version control system.  
# MAGIC
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC
# MAGIC Unity catalog allows administrators to define catalogs to represent the resources to be accessed from different environments. For example, production systems can run from a production catalog while data scientists and engineers can work in a development catalog. Models in any catalog can be given an `Alias` that signifies which version of a model is currently the latest one for a particular use. Common aliases in clude `Champion` for the current production model while `Challenger` might be used to indicate a new candidate model undergoing testing & review.
# MAGIC
# MAGIC Users with appropriate permissions can transition models between catalogs and aliases.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sending our model to the Unity Catalog
# MAGIC
# MAGIC We'll programatically select the best model from our last Auto-ML run and deploy it in the registry. We can easily do that using MLFlow `search_runs` API:

# COMMAND ----------

#Use Databricks Unity Catalog to save our model
import mlflow
from mlflow.tracking.client import MlflowClient
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# Let's get the latest experiment ID 
xp_path = f"/Users/{current_user}/databricks_automl/{dbName}"
filter_string=f"name LIKE '{xp_path}/%'"
automl_experiment_id = mlflow.search_experiments(filter_string=filter_string,order_by=["last_update_time DESC"])[0].experiment_id
print(f"Found AutoML experiment id: {automl_experiment_id}")

# COMMAND ----------

# Optional: Load MLflow Experiment and see all runs
df = spark.read.format("mlflow-experiment").load(automl_experiment_id)
display(df)

# COMMAND ----------

# Get our the best model run (search run with highest f1 score)
best_model = mlflow.search_runs(experiment_ids=[automl_experiment_id], order_by=["metrics.val_f1_score DESC"], max_results=1, filter_string="status = 'FINISHED'")
run_id = best_model.iloc[0]['run_id']
best_model

# COMMAND ----------

# MAGIC %md Once we have our best model, we can now deploy it in production using it's run ID

# COMMAND ----------

#deploy model in UC
model_name = f"churn_model_{current_user_no_at}"
src_model_uri = f"runs:/{run_id}/model"
dest_model_path = f"{dev_catalog}.{dbName}.{model_name}"
model_alias = "Challenger"
tablename = 'dbdemos_mlops_churn_features'
table_path = f"{dev_catalog}.{dbName}.{tablename}"

#add some tags that we'll reuse later to validate the model
client = mlflow.tracking.MlflowClient()
client.set_tag(run_id, key='demographic_vars', value='senior_citizen,gender_female')
client.set_tag(run_id, key='db_table', value=table_path)

#Add model within our catalog
model_details = mlflow.register_model(src_model_uri, dest_model_path)
# Flag it as Production ready using UC Aliases
client.set_tag(run_id, key='db_table', value=table_path)
client.set_registered_model_alias(name=dest_model_path, alias=model_alias, version=model_details.version)
# refresh the details
model_details = client.get_model_version_by_alias(dest_model_path, model_alias)

# COMMAND ----------

model_details

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Description
# MAGIC We'll do this for the registered model overall, and the particular version.

# COMMAND ----------

#The main model description, typically done once.
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a customer will churn.  It is used to update the Telco Churn Dashboard in DB SQL."
)

#Gives more details on this specific model version
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using XGBoost. Eating too much cake is the sin of gluttony. However, eating too much pie is okay because the sin of pie is always zero."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next: MLOps model testing and validation
# MAGIC
# MAGIC Next: Find out how the model is being tested before promotion to production [using the Databricks Staging test notebook]($./04_model_validation)

# COMMAND ----------


