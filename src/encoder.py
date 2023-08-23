import cv2
import numpy as np
from torchvision.transforms import transforms
from ultralytics import YOLO
from utils.transform import *
from utils.process import *
from utils.task import *
from utils.to_df_post import *
from ultralytics.nn.tasks import *
import yaml
import sys
sys.path.append('../../')



yaml_file = open("../config/cfg.yaml")
cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
models = cfg['model']['nano']
lst_img = cfg['data']['img_encode']
lst_vid = cfg['data']['vid_encode']
time_cut = cfg['encoder']['time_video_cut']
jsonpath = cfg['encoder']['rs_json_path']

def main():
    #encoder image
    df_feature_img = encode_image(model = models,lst_image = lst_img)
    df_insta = to_df_encoder(jsonfile = jsonpath 
                             ,df_feature = df_feature_img,segment = 'insta')
    #encoder videos
    df_feature_vid= encode_video(model = models, lst_video = lst_vid
                                ,timing_cut = time_cut)
    df_tiktok = to_df_encoder(jsonfile = jsonpath 
                             ,df_feature = df_feature_vid,segment = 'tiktok')
    df = concat_df(df_insta,df_tiktok)
    print(df)
if __name__ == "__main__":
    main()