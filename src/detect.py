from utils.train import *
from utils.val import *
import yaml
import sys
sys.path.append('../../')
import logging.config

#Load yaml file:
yaml_file = open("../config/cfg.yaml")
cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
# train config:
model = cfg['model']['nano'] #change model here
cfg_data = cfg['data']['config']
imgsz = cfg['train']['imgsz']
epochs = cfg['train']['epochs']
batchsz = cfg['train']['batchsz']
save_model = cfg['train']['name']
# val config
model_best = cfg['model']['best_weight']
batchsz_val = cfg['val']['batchsz']

def main():
    rs_train = train(model,data = cfg_data, imgsz = imgsz,
                     epochs = epochs, batch = batchsz, save_model=save_model)
    print(rs_train)
    print('-------------------------------------------')
    rs_val = val(model_val=model_best, data=cfg_data,
                 imgsz=imgsz, batch=batchsz_val)
    print(rs_val)

if __name__ == "__main__":
    main()
    