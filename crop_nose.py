import os, glob, json
from PIL import Image

anno_file = '/media/irelin/data_disk/dataset/afp/noseprint_recognition/result.json'
image_root = '/media/irelin/data_disk/dataset/afp/noseprint_recognition/2.sampling'
output_dir = '/media/irelin/data_disk/dataset/afp/noseprint_recognition/nose_split'

annos = json.load(open(anno_file))

for i, anno in enumerate(annos):
    im_fn = os.path.basename(anno['data']['image']).split('-')[1]
    im_dir = "_".join(im_fn.split("_")[:2])
    im_path = os.path.join(image_root, im_dir, im_fn)
    print(i, len(annos), im_path)
    if not os.path.isfile(im_path):
        print("not exists")
        continue
    anno_w = anno["completions"][0]["result"][0]["original_width"]
    anno_h = anno["completions"][0]["result"][0]["original_height"]
    im = Image.open(im_path)
    w, h = im.size
    if w != anno_w or h != anno_h:
        print('not same size')
        continue

    bbox_w = float(anno["completions"][0]["result"][0]["value"]["width"])
    bbox_h = float(anno["completions"][0]["result"][0]["value"]["height"])
    x1 = float(anno["completions"][0]["result"][0]["value"]["x"])
    y1 = float(anno["completions"][0]["result"][0]["value"]["y"])
    x2 = x1 + bbox_w
    y2 = y1 + bbox_h

    x1 = x1 / 100 * w
    x2 = x2 / 100 * w
    y1 = y1 / 100 * h
    y2 = y2 / 100 * h

    im = im.crop((x1, y1, x2, y2))
    os.makedirs(os.path.join(output_dir, im_dir), exist_ok=True)
    output_path = os.path.join(output_dir, im_dir, os.path.splitext(im_fn)[0] + ".jpg")
    im.save(output_path, quality=100)
