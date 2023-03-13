import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.get_experiment(experiment_id=1)
    sample = mlflow.search_runs(experiment_ids=runs.experiment_id)
    
    max_accuracy = max(sample["metrics.accuracy"])
    max_accuracy_run_id = list(sample[sample["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    print(max_accuracy_run_id)

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        current_version = mv["version"]
        logged_model = mv["source"]
        pprint(mv, indent=4)
        client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")

        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(loaded_model)
        joblib.dump(loaded_model, model_dir)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
