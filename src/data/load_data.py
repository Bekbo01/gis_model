import yaml
import argparse
import numpy as np 
import pandas as pd 

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
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data(data_path,model_var):
    """
    load csv dataset from given path
    input: csv path 
    output:pandas dataframe 
    note: Only 6 variables are used in this model building stage for the simplicity.
    """
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    df = df[model_var]
    #df.replace(df.get('NULL'), np.nan, inplace=True)
    df.fillna(value=np.nan, inplace=True)
    df_list = []
    for well in  df['WELL'].unique().tolist():
        data_local = df[df['WELL']==well]
        if data_local.isna().values.any():
            data_local = removing_long_nan_pause(data_local, 0.2, 'DEPT')
        df_list.append(data_local)
    df_new = pd.concat(objs=df_list)
    return df_new

def load_raw_data(config_path):
    """
    load data from external location(data/external) to the raw folder(data/raw) with train and teting dataset 
    input: config_path 
    output: save train file in data/raw folder 
    """
    config=read_params(config_path)
    external_data_path=config["external_data_config"]["external_data_csv"]
    raw_data_path=config["raw_data_config"]["raw_data_csv"]
    model_var=config["raw_data_config"]["model_var"]
    
    df=load_data(external_data_path,model_var)
    df.to_csv(raw_data_path,index=False)
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)