import os
import argparse
import glob
import json
import random
from multiprocessing import Pool
import traceback
from PIL import Image
from dataset.online_dataset import OnlineFontDataset
from trainer import gen_params, transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_dir', type=str, default="/home/irelin/resource/font_recognition/val_dataset")

    parser.add_argument('-f', '--font_dir', type=str, default="/home/irelin/resource/font_recognition/font")
    parser.add_argument('--font_list', type=str, default="./db/val_font_list.json")
    parser.add_argument('--bg_dir', type=str, default="/home/irelin/resource/font_recognition/val_bg")

    parser.add_argument('--train_dataset_param_func', type=str, default='get_train_params',  # get_params_noraml
                        required=False)  # func or gen_param.json
    parser.add_argument('--train_transform_func_name', type=str, default='get_train_transforms',
                        # get_train_transforms_normal
                        required=False)  # func or gen_param.json

    parser.add_argument('--num_sample_each_class', type=int, default=10)
    parser.add_argument('--min_num_chars', type=int, default=1)
    parser.add_argument('--max_num_chars', type=int, default=6)

    parser.add_argument('--han_unicode_file', type=str, default="db/union_korean_unicodes.json")
    parser.add_argument('--eng_unicode_file', type=str, default="db/eng_unicodes.json")
    parser.add_argument('--num_unicode_file', type=str, default="db/number_unicodes.json")
    parser.add_argument('--han_prob', type=float, default=0.4)
    parser.add_argument('--eng_prob', type=float, default=0.3)
    parser.add_argument('--num_prob', type=float, default=0.3)
    parser.add_argument('--mix_prob', type=float, default=0.25)
    parser.add_argument('--simple_img_prob', type=float, default=0.2)

    parser.add_argument('--font_size_range', type=str, default='10,220')

    parser.add_argument('--same_text_in_batch_prob', default=0., type=float)
    parser.add_argument('--same_font_size_in_batch_prob', default=0., type=float)
    parser.add_argument('--same_text_params_in_batch_prob', default=0., type=float)
    parser.add_argument('--use_text_persp_trans_prob', default=0.1, type=float)
    parser.add_argument('--use_img_persp_trans_prob', default=0.4, type=float)

    parser.add_argument('-w', '--num_processes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    random.seed(args.seed)

    bg_list = glob.glob(os.path.join(args.bg_dir, "*"))

    generation_params = getattr(gen_params, args.train_dataset_param_func)()

    train_transform_func = getattr(transforms, args.train_transform_func_name)
    train_transforms = train_transform_func(use_online=False)

    han_unicodes = json.load(open(args.han_unicode_file))
    eng_unicodes = json.load(open(args.eng_unicode_file))
    num_unicodes = json.load(open(args.num_unicode_file))
    font_size_range = args.font_size_range.split(",")
    font_size_range = list(range(int(font_size_range[0]), int(font_size_range[1]) + 1))

    font_list = glob.glob(os.path.join(args.font_dir, "*"))
    font_list.sort()

    feed_data = []
    for i in range(len(font_list)):
        font_name = os.path.splitext(os.path.basename(font_list[i]))[0]
        class_dir = os.path.join(args.output_dir, font_name)
        file_cnt = len(glob.glob(os.path.join(class_dir, "*")))
        if file_cnt >= args.num_sample_each_class:
            print(font_name, "skip")
            continue
        for j in range(args.num_sample_each_class):
            feed_data.append([i, j])

    dataset = OnlineFontDataset(font_list, transform=train_transforms, generation_params=generation_params,
                                bg_list=bg_list,
                                num_sample_each_class=args.num_sample_each_class,
                                min_chars=args.min_num_chars, max_chars=args.max_num_chars,
                                hangul_unicodes=han_unicodes, eng_unicodes=eng_unicodes,
                                number_unicodes=num_unicodes,
                                hangul_prob=args.han_prob, eng_prob=args.eng_prob,
                                num_prob=args.num_prob, mix_prob=args.mix_prob,
                                simple_img_prob=args.simple_img_prob,
                                font_size_range=font_size_range,
                                same_text_in_batch_prob=args.same_text_in_batch_prob,
                                same_font_size_in_batch_prob=args.same_font_size_in_batch_prob,
                                same_text_params_in_batch_prob=args.same_text_params_in_batch_prob,
                                use_text_persp_trans_prob=args.use_text_persp_trans_prob,
                                use_img_persp_trans_prob=args.use_img_persp_trans_prob,
                                skip_exception=True,
                                use_debug=True,
                                change_font_in_error=False,
                                use_random_idx=False
                                )


    def _main(feed_item):
        font_idx = feed_item[0]
        text_idx = feed_item[1]
        font_path = font_list[font_idx]
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        class_dir = os.path.join(args.output_dir, font_name)
        print(font_idx, text_idx, font_name)
        os.makedirs(class_dir, exist_ok=True)
        tried = 0
        try:
            im = None
            while True:
                try:
                    dataset._sampling_text()
                    im = dataset.create_text_image(font_idx)
                    break
                except Exception as e:
                    tried += 1
                    if tried > 6:
                        raise e
                    pass

            # im, _ = dataset.__getitem__(font_idx)
            # print("debug", _, font_idx)
            if len(glob.glob(os.path.join(class_dir, "*"))) >= args.num_sample_each_class:
                print(font_name, "skip")
                return False
            fp = os.path.join(class_dir, "%09d.jpg" % text_idx)
            while os.path.isfile(fp):
                text_idx += 1
                fp = os.path.join(class_dir, "%09d.jpg" % text_idx)
            Image.fromarray(im).save(os.path.join(class_dir, "%09d.jpg" % text_idx), quality=100)
        except Exception:
            print("gen error", font_name, font_idx)
            traceback.print_exc()


    with Pool(args.num_processes) as pool:  # ThreadPool(8) as pool:
        print("start multi processing")
        pool.map(_main, feed_data)

    print("end of making")
