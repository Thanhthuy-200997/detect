import cv2
import glob

def read_image(image):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    return img

def cut_save_frame_vid(vid,sc):
    # parse_vid = vid.split("/")
    # lenght = len(parse_vid)
    # name = parse_vid[lenght - 1].split(".")[0]
    vidcap = cv2.VideoCapture(vid)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print(fps)

    while success:
        success,image = vidcap.read()
        # print('read a new frame:',success)
        if count%(sc*fps) == 0 :
            img_name = vid + "_" + str(count) + ".jpg"
            cv2.imwrite(img_name,image)
            print(f'successfully cut {vid} to image {img_name}')
        count+=1
        
def loop_cut_video(path_video,sc):
    path_vid = path_video + "/" + "*.mp4"
    vid = glob.glob(path_vid)
    for vd in vid:
        print('Start cut and save  %s video to frame'%vd)
        cut_save_frame_vid(vd,sc)
    return None
    

def resize(image,width,height):
    dim = (width,height)
    # img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
    return img