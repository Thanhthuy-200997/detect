import pandas as pd
import re
from utils.prepare_dataframe import *

def create_df_post(jsonfile,segment):
    if segment == 'insta':
        print(f'---------- Create dataframe from instagram path: {jsonfile} ----------')
        jsonfile = jsonfile + '/post_meta.json'
        data = load_json(jsonfile)
        temp = pd.DataFrame(data['post_meta'])
        df = temp[['mediaid','insta_url','links']].iloc[1:,:]
        df = df.explode('links')
        df['parse_name'] = df['links'].apply(lambda x: re.findall("//.*/(.*)\?", x))
        df['img_name'] = df['parse_name'].apply(lambda x: ''.join(x))
        print(f'---------- Create dataframe from instagram path {jsonfile} complete----------')
    elif segment == 'tiktok':
        print(f'---------- Create dataframe from tiktok path {jsonfile} ----------')
        jsonfile = jsonfile + '/video_meta.json'
        data = load_json(jsonfile)
        temp = pd.DataFrame(data['video_meta'])
        df = temp[['account','meta_post']].iloc[1:,:]
        df['links'] = df['meta_post'].apply(lambda x: x.keys())
        df = df.explode('links')
        df['img_name'] = df['links'].apply(lambda x: x.split("/")[-1]) 
        print(f'---------- Create dataframe from tiktok path {jsonfile} complete----------')
    else:
        print('Please check segment. Must be insta or tiktok!!!')             
    return df

def to_df_encoder(jsonfile,df_feature,segment):
    '''
        from_json: from path json data which in result json crawl data
        to_json: destination path json in detect
    '''
    # create_file(from_json,to_json)
    df_post= create_df_post(jsonfile,segment)
    df = merge_df(df_post,df_feature)
    df = df[['links','img_name','vector']]
    return df
