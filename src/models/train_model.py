import yaml
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from sklearn.metrics import *
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = [0,1,2,3,4   ]
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop([target, 'WELL', 'DEPT'], axis=1)
    y=df[[target]]
    return x,y    

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    #accuracy = accuracy_score(actual, pred)
    return (rmse, mae, r2) #, accuracy)

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["xgboost"]["max_depth"]
    n_estimators=config["xgboost"]["n_estimators"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

################### MLFLOW ##########################################################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    #mlflow.xgboost.autolog()
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        """
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.005,
            random_state=42,
            seed=42,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=1,
            gamma=1)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, y_pred)
        """
        """
        paramters = {"n_estimators"   :[10,15,25,50,100,150],
             "max_depth"      :[3,5,7,10],
             "loss"           :["ls", "lad"]
        }
        # define the grid search and optimization metric
        grid = GridSearchCV(estimator=XGBRegressor(),
                            param_grid=paramters,
                            scoring="r2",
                            cv=5,
                            n_jobs=-1)
        """
        pipe_steps = [
            ('scaler', MinMaxScaler()),
            ('XGBClassifier', XGBClassifier(base_score=0.5, booster='gbtree',
                                colsample_bylevel=1, colsample_bynode=1,
                                colsample_bytree=1, eval_metric='mlogloss',
                                gamma=0, gpu_id=-1, importance_type='gain',
                                interaction_constraints='', learning_rate=0.3,
                                max_delta_step=0, max_depth=max_depth,
                                min_child_weight=1,
                                monotone_constraints='()', n_estimators=n_estimators,
                                n_jobs=-1, num_parallel_tree=1,
                                objective='multi:softprob', random_state=42,
                                reg_alpha=0.2, reg_lambda=1,
                                scale_pos_weight=None, subsample=0.8,
                                tree_method='exact', validate_parameters=1,
                                verbosity=None))
            ]
        model = Pipeline(pipe_steps)
        # perform the grid search
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        r2 = r2_score(test_y, y_pred)
        mse = mean_squared_error(test_y, y_pred) 
        #accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')
        #accuracy = roc_auc_score(test_y, y_pred)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("mse", mse)
        #mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        #mlflow.log_metric("accuracy", auc_score)
       
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

