import cv2
import numpy as np
from utils.prepare_dataframe import *
from utils.to_df_post import *
from ultralytics import YOLO
from utils.transform import *
from utils.process import *
from ultralytics.nn.tasks import *
from PIL import Image


def encode(models,image):
    t = Transform()
    model = YOLO(models)
    img = Image.open(image)
    input = t.transform(img)
    pred = BaseModel._predict_once(model.model,input)
    out = pred[0].detach().numpy()
    # print(out)
    img.close()
    return out

def encode_image(model,lst_image):
    lst_img_name = []
    lst_encode_img = []
    print(f'---- Encoder image from instagram {lst_image} ----')
    lst_img = glob.glob(lst_image)
    for image in lst_img:
        im = image.split("\\")[-1]
        lst_img_name.append(im)
        img_out = encode(model,image)
        lst_encode_img.append(img_out)
    print('---- Successful encoder image from instagram')
    df_feature = create_dataframe(lst_img_name,lst_encode_img)
    return df_feature
        
def encode_video(model,lst_video,timing_cut):
    lst_img_name_from_vid = []
    lst_encode_img_from_vid = []
    print(f'---- Cut video in {lst_video} to image ----')
    loop_cut_video(lst_video,timing_cut)
    print(f'---- Successful cut video to image  ----')
    lst_image_from_vid = lst_video + "/" + "*.jpg"
    lst_image_vid = glob.glob(lst_image_from_vid)
    print(f'---- Encoder video from tiktok {lst_image_vid} ----')
    for img_form_vid in lst_image_vid:
        im = img_form_vid.split("\\")[-1].split(".")[0]
        lst_img_name_from_vid.append(im)
        img_from_video_out = encode(model,img_form_vid)
        lst_encode_img_from_vid.append(img_from_video_out)
    print('---- Successful encoder video from tiktok')
    df_feature = create_dataframe(lst_img_name_from_vid,lst_encode_img_from_vid)
    return df_feature
    
        
