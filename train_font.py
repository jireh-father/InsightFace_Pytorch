from config import get_config
from Learner import face_learner
import argparse
import random
import glob
import os
import json
import gen_params, transforms
from data.online_dataset import OnlineFontDataset
from data.hook_dataloader import HookDataLoader
from pathlib import Path

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for common image metric learning')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)

    parser.add_argument("--embedding_size", help="embedding_size", default=512, type=int)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='common',
                        type=str)

    parser.add_argument("--max_positive_cnt", default=1000, type=int)
    parser.add_argument('--pin_memory', default=False, action="store_true")
    parser.add_argument("--val_batch_size", default=256, type=int)
    parser.add_argument("--val_pin_memory", default=False, action='store_true')
    parser.add_argument("--not_use_pos", default=False, action='store_true')
    parser.add_argument("--not_use_neg", default=False, action='store_true')

    parser.add_argument('--work_path', type=str, default=None, required=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--benchmark', default=False, action="store_true")

    parser.add_argument('--font_dir', type=str, default=None, required=False)
    parser.add_argument('-f', '--font_list', type=str, default='./db/train_font_list.json', required=False)
    parser.add_argument('--bg_dir', type=str, default='resource/train_bg', required=False)
    parser.add_argument('--val_img_dirs', type=str,
                        default='{"mix":"","kr":"","eng":"","num":""}',
                        required=False)

    parser.add_argument('--train_dataset_param_func', type=str, default='get_train_params',  # get_params_noraml
                        required=False)  # func or gen_param.json
    parser.add_argument('--train_transform_func_name', type=str, default='get_train_transforms',
                        # get_train_transforms_normal
                        required=False)  # func or gen_param.json
    parser.add_argument('--val_transform_func_name', type=str, default='get_test_transforms',
                        # get_train_transforms_normal
                        required=False)  # func or gen_param.json
    parser.add_argument('--num_sample_each_class', type=int, default=1000)

    parser.add_argument('--min_num_chars', type=int, default=1)
    parser.add_argument('--max_num_chars', type=int, default=10)

    parser.add_argument('--input_size', type=int, default=112)

    parser.add_argument('--use_random_crop', default=False, action="store_true")
    parser.add_argument('--use_gray', default=False, action="store_true")
    parser.add_argument('--use_same_random_crop_in_batch', default=False, action="store_true")

    parser.add_argument('--same_text_in_batch_prob', default=1., type=float)
    parser.add_argument('--same_font_size_in_batch_prob', default=1., type=float)
    parser.add_argument('--same_text_params_in_batch_prob', default=1., type=float)
    parser.add_argument('--use_text_persp_trans_prob', default=0.1, type=float)
    parser.add_argument('--use_img_persp_trans_prob', default=0.3, type=float)

    parser.add_argument('--han_unicode_file', type=str, default="db/union_korean_unicodes.json")
    parser.add_argument('--eng_unicode_file', type=str, default="db/eng_unicodes.json")
    parser.add_argument('--num_unicode_file', type=str, default="db/number_unicodes.json")
    parser.add_argument('--han_prob', type=float, default=0.4)
    parser.add_argument('--eng_prob', type=float, default=0.3)
    parser.add_argument('--num_prob', type=float, default=0.3)
    parser.add_argument('--mix_prob', type=float, default=0.5)
    parser.add_argument('--simple_img_prob', type=float, default=0.3)

    parser.add_argument('--font_size_range', type=str, default='10,220')
    parser.add_argument('--dataset_debug', default=False, action="store_true")

    parser.add_argument('--only_use_pixel_transform', default=False, action="store_true")
    parser.add_argument('--use_blur', default=False, action="store_true")
    parser.add_argument('--use_flip', default=False, action="store_true")
    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--pooling', default='GeM', type=str)
    parser.add_argument('--last_fc_dropout', type=float, default=0.0)
    parser.add_argument('--pretrained', default=False, action="store_true")
    parser.add_argument('--loss_module', default='arcface', type=str)

    parser.add_argument('--s', type=float, default=30.0)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--ls_eps', type=float, default=0.0)
    parser.add_argument('--theta_zero', type=float, default=1.25)

    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--restore_suffix', default=None, type=str)

    args = parser.parse_args()
    conf = get_config()

    for arg in vars(args):
        print(arg, getattr(args, arg))
        setattr(conf, arg, getattr(args, arg))

    conf.work_path = Path(conf.work_path)
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    if args.seed is not None:
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.benchmark:
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        cudnn.deterministic = True

    bg_list = glob.glob(os.path.join(args.bg_dir, "*"))

    generation_params = getattr(gen_params, args.train_dataset_param_func)()

    if hasattr(transforms, args.train_transform_func_name):
        train_transform_func = getattr(transforms, args.train_transform_func_name)
        train_transforms = train_transform_func(args.input_size, args.use_random_crop, args.use_gray,
                                                use_same_random_crop_in_batch=args.use_same_random_crop_in_batch,
                                                only_use_pixel_transform=args.only_use_pixel_transform,
                                                use_flip=args.use_flip, use_blur=args.use_blur
                                                )
    else:
        train_transforms = transforms.get_simple_transforms(input_size=args.input_size,
                                                            use_random_crop=args.use_random_crop,
                                                            use_same_random_crop_in_batch=args.use_same_random_crop_in_batch,
                                                            use_gray=args.use_gray)

    han_unicodes = json.load(open(args.han_unicode_file))
    eng_unicodes = json.load(open(args.eng_unicode_file))
    num_unicodes = json.load(open(args.num_unicode_file))
    font_size_range = args.font_size_range.split(",")
    font_size_range = list(range(int(font_size_range[0]), int(font_size_range[1]) + 1))

    font_list = [os.path.join(args.font_dir, font_name) for font_name in json.load(open(args.font_list))]
    font_list.sort()
    num_classes = len(font_list)
    conf.num_classes = num_classes

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
                                input_size=args.input_size,
                                use_same_random_crop_in_batch=args.use_same_random_crop_in_batch,
                                use_debug=args.dataset_debug
                                )

    train_loader = HookDataLoader(dataset, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory,
                                  batch_size=args.batch_size)

    val_transform_func = getattr(transforms, args.val_transform_func_name)
    val_transforms = val_transform_func(input_size=args.input_size, use_gray=args.use_gray)
    learner = face_learner(conf, val_transforms=val_transforms, train_loader=train_loader)

    learner.train(conf, args.epochs)
