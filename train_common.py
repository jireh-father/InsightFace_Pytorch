from config import get_config
from Learner import face_learner
import argparse
import random
import transforms
from pathlib import Path

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for common image metric learning')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)

    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='common',
                        type=str)

    parser.add_argument("--embedding_size", help="embedding_size", default=512, type=int)

    parser.add_argument("-t", "--train_img_dir", default=None, type=str)

    parser.add_argument("--max_positive_cnt", default=1000, type=int)
    parser.add_argument("--val_batch_size", default=256, type=int)
    parser.add_argument('--pin_memory', default=False, action="store_true")
    parser.add_argument("--val_pin_memory", default=False, action='store_true')
    parser.add_argument("--not_use_pos", default=False, action='store_false')
    parser.add_argument("--not_use_neg", default=False, action='store_false')

    parser.add_argument('--work_path', type=str, default=None, required=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--benchmark', default=False, action="store_true")

    parser.add_argument('--val_img_dirs', type=str,
                        default='{"val":"path"}',
                        required=False)
    parser.add_argument('--train_transform_func_name', type=str, default='get_train_common_transforms',
                        # get_train_transforms_normal
                        required=False)  # func or gen_param.json
    parser.add_argument('--val_transform_func_name', type=str, default='get_val_common_transforms',
                        # get_train_transforms_normal
                        required=False)  # func or gen_param.json

    parser.add_argument('--input_size', type=int, default=112)

    parser.add_argument('--use_random_crop', default=False, action="store_true")
    parser.add_argument('--use_gray', default=False, action="store_true")

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

    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--no_transforms', default=False, action="store_true")

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

    train_transform_func = getattr(transforms, args.train_transform_func_name)
    train_transforms = train_transform_func(args.input_size, args.use_random_crop, args.use_gray,
                                            only_use_pixel_transform=args.only_use_pixel_transform,
                                            use_flip=args.use_flip, use_blur=args.use_blur,
                                            no_transforms=args.no_transforms
                                            )

    val_transform_func = getattr(transforms, args.val_transform_func_name)
    val_transforms = val_transform_func(input_size=args.input_size, use_gray=args.use_gray)

    learner = face_learner(conf, train_transforms=train_transforms, val_transforms=val_transforms)

    learner.train(conf, args.epochs)
