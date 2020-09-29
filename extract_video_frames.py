import cv2
import os
import glob

video_dir = '/media/irelin/data_disk/dataset/afp/noseprint_recognition/videos'
output_dir = '/media/irelin/data_disk/dataset/afp/noseprint_recognition/frames'
for i, vpath in enumerate(glob.glob(os.path.join(video_dir, "*"))):
    tmp_output_dir = os.path.join(output_dir, str(i))
    os.makedirs(tmp_output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(vpath)
    success, image = vidcap.read()
    count = 0
    print(vpath)
    while success:
        cv2.imwrite(os.path.join(tmp_output_dir, "frame%d.jpg" % count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
