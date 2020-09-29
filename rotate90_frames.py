import glob
import os
from PIL import Image, ExifTags
import traceback

root_dir = "/media/irelin/data_disk/dataset/afp/noseprint_recognition/frames"
image_dirs = glob.glob(os.path.join(root_dir, "*"))

for image_dir in image_dirs:
    for im_path in glob.glob(os.path.join(image_dir, "*")):
        try:
            print(im_path)
            image = Image.open(im_path)
            image = image.rotate(-90, expand=True)
            image.save(im_path, quality=100)
            image.close()
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            traceback.print_exc()
            print("skip")
            pass
