import shutil
import os
import pandas as pd
import json
from distutils.dir_util import copy_tree

def create_file(from_json,to_json):
    shutil.rmtree(to_json, ignore_errors=True)
    copy_tree(from_json, to_json)
    return None

def load_json(jsondata):
    with open(jsondata, 'r',encoding='utf8') as f:
        data = json.load(f)
    f.close()
    return data

def create_dataframe(lst_img_name,lst_vector):
    extract_feature = pd.DataFrame(columns = ['img_name','vector'])
    extract_feature['img_name'] = lst_img_name
    extract_feature['vector'] = lst_vector
    return extract_feature

def merge_df(df,df_feature):
    df_merge = pd.merge(df, df_feature, on='img_name', how='inner')
    return df_merge

def concat_df(df1,df2):
    df_concat = pd.concat([df1, df2], axis=0).reset_index(drop = True)
    return df_concat

    


