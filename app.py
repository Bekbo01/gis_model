from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib 
import pandas as pd
from sklearn.metrics import r2_score
import csv

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

class  NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def removing_long_nan_pause(data, step, dept_label, interp_method='linear', nan_long_label=5):
    created_df = pd.DataFrame([])
    dframes = []
    local_df = data.copy()
    local_df = local_df[local_df.isna().any(axis=1)]  # taking only rows with nan
    local_df['delta'] = local_df[dept_label].diff(1)
    local_df['delta'].replace(np.nan, step, inplace=True)

    ind = local_df.last_valid_index()
    df_cr = local_df[local_df.delta > step+0.1]

    if len(local_df) <= nan_long_label:
        local_df2 = local_df  # if short pause, then left it
        # print('only one short pause with length less than or equal to 5 rows')

    elif df_cr.shape[0] == 0 and len(local_df) > nan_long_label:
        local_df2 = local_df.dropna() 
        # print('only one short pause with the length greater than 5 rows')

    # if there sre some long pauses
    # 1 long pause
    elif df_cr.shape[0] == 1:
        n = df_cr.index.item() 
        k = ind - n
        if n > nan_long_label and k > nan_long_label:
            local_df2 = local_df.dropna()
        elif n > nan_long_label and k <= nan_long_label:
            local_df2 = local_df.drop(local_df.index[:n])
        elif n <= nan_long_label and k > nan_long_label:
            local_df2 = local_df.drop(local_df.index[n:])
        else:
            local_df2 = local_df

    elif df_cr.shape[0] > 1:
        L = df_cr.index.to_list()
        ind = local_df.index.to_list()[len(local_df) - 1]
        L.append(ind + 1)
        local_df2 = pd.DataFrame([])
        last_check = 0
        dfs = []
        for i in L:
            local = local_df.loc[last_check:i - 1]
            if len(local) <= nan_long_label:
                dfs.append(local)
            last_check = i
        local_df2 = pd.concat(objs=dfs)

    else:
        # print('something get wrong')
        return None

    dframes.append(local_df2)

    created_df = pd.concat(objs=dframes)  # created df only with short pause (indexes are saved, as in original df)
    df_raw = data.dropna()
    df_new = pd.concat(objs=[df_raw, created_df])  # concatinating df without NaN with short pause
    if interp_method=='linear':
        df_new = df_new.interpolate(method='linear', axis=0)  # interpolating short pause
    elif interp_method=='pad':
        df_new = df_new.interpolate(method='pad', axis=0) 
    else:
        df_new = df_new.dropna()  
        # interpolating short pause
    df_new = df_new.drop(columns='delta')

    return df_new 


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = joblib.load(model_dir_path)
    x=data.drop(['Facies_IPSOM_IPSOM', 'WELL', 'DEPT'], axis=1)
    x = x.astype(float)
    print('tipa ozgerttim')
    y=data[['Facies_IPSOM_IPSOM']]
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    return r2 

def validate_input(df):
    df.replace('', np.nan, inplace=True)
    df['DEPT'] = df['DEPT'].astype(float)
    return removing_long_nan_pause(df, 0.2, 'DEPT')

def form_response(df):
    df = validate_input(df)
    if not df.empty:
        response = predict(df)
        return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files['upload_file']
            file = file.read().decode('UTF-8')
            csv_dicts = [{k: v for k, v in row.items()} for row in csv.DictReader(file.splitlines(), skipinitialspace=True)]
            df = pd.DataFrame(csv_dicts)
            if '' in df.columns:
                df.drop('',axis=1, inplace=True)
            response = form_response(df)
            return render_template("index.html", response=response)
        except Exception as e:
           print(e)
           error = {"error": "Something went wrong!! Try again later!"}
           error = {"error": e}
           return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)