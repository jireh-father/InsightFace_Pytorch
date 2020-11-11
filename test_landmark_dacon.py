from config import get_config
from Learner import face_learner
import argparse
import random
import transforms
from pathlib import Path
import torch
from PIL import Image
import os
import glob
import csv
from torch.nn import functional as F


# python train.py -net mobilefacenet -b 200 -w 4
class CustomDataset(torch.utils.data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for image_dir in glob.glob(os.path.join(root, "*")):
            for image_file in glob.glob(os.path.join(image_dir, "*")):
                self.samples.append(image_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path = self.samples[index]
        sample = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        return sample, os.path.splitext(os.path.basename(path))[0]

    def __len__(self):
        return len(self.samples)


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

    parser.add_argument("-t", "--test_img_dir", default=None, type=str)
    parser.add_argument("--label_file", default=None, type=str)

    parser.add_argument("--max_positive_cnt", default=1000, type=int)
    parser.add_argument("--val_batch_size", default=256, type=int)
    parser.add_argument('--pin_memory', default=False, action="store_true")
    parser.add_argument("--not_use_pos", default=False, action='store_true')
    parser.add_argument("--not_use_neg", default=False, action='store_true')

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
    parser.add_argument('--use_center_crop', default=False, action="store_true")
    parser.add_argument('--center_crop_ratio', default=0.8, type=float)
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

    parser.add_argument('--ft_model_path', default=None, type=str)
    parser.add_argument('--no_strict', default=False, action="store_true")

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

    transforms = transforms.get_test_transforms_v2(args.input_size, use_crop=args.use_crop,
                                                   center_crop_ratio=args.center_crop_ratio, use_gray=args.use_gray)
    dataset = CustomDataset(args.test_img_dir, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    learner = face_learner(conf, inference=True)

    for imgs, labels in self.loader:
        imgs = imgs.to(conf.device)
        labels = labels.to(conf.device)
        self.optimizer.zero_grad()
        embeddings = self.model(imgs)
        thetas = self.head(embeddings, labels)

    device = 'cuda'
    total_scores = []
    total_indices = []
    total_file_names = []
    for step, (imgs, file_names) in enumerate(dataloader):
        if step > 0 and step % args.log_step_interval == 0:
            print(step, len(dataloader))
        imgs = imgs.to(device)
        total_file_names += list(file_names)
        with torch.set_grad_enabled(False):
            embeddings = learner.model(imgs)
            outputs = learner.head(embeddings, labels)
            scores, indices = torch.max(F.softmax(outputs, 1), dim=1)
            total_indices += list(indices.cpu().numpy())
            total_scores += list(scores.cpu().numpy())

    rows = zip(total_file_names, total_indices, total_scores)
    output_dir = os.path.dirname(args.output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "landmark_id", "conf"])
        for row in rows:
            writer.writerow(row)
    print("done")
