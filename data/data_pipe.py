from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle

from tqdm import tqdm
import random


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def get_train_dataset(imgs_folder, train_transforms=None):
    if not train_transforms:
        train_transforms = trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    ds = ImageFolder(imgs_folder, train_transforms)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_custom_train_dataset(imgs_folder, train_transforms=None):
    class CustomDataset(ImageFolder):
        """__init__ and __len__ functions are the same as in TorchvisionDataset"""

        def __init__(self, root, transform=None):
            super(CustomDataset, self).__init__(root, transform=transform)

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """

            while True:
                try:
                    path, target = self.samples[index]
                    sample = np.array(self.loader(path))
                    if self.transform is not None:
                        sample = self.transform(image=sample)['image']
                    return sample, target
                except Exception as e:
                    # traceback.print_exc()
                    print(str(e), path)
                    index = random.randint(0, len(self) - 1)

    ds = CustomDataset(imgs_folder, train_transforms)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_dacon_landmark_train_dataset(imgs_folder, train_transforms=None, label_file=None):
    import torch
    import json
    import os
    import glob

    class CustomDataset(torch.utils.data.Dataset):
        """__init__ and __len__ functions are the same as in TorchvisionDataset"""

        def __init__(self, root, transform=None, label_file=None):
            self.transform = transform
            labels = json.load(open(label_file))
            label_map = {}
            for label in labels['categories']:
                label_map[label['landmark_name']] = label['landmark_id']

            samples = []
            for dir in glob.glob(os.path.join(root, '*')):
                label_name = os.path.basename(dir)
                label_idx = label_map[label_name]
                for path in glob.glob(os.path.join(dir, "*")):
                    samples.append([path, label_idx])
            self.samples = samples

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """

            while True:
                try:
                    path, target = self.samples[index]
                    sample = np.array(Image.open(path).convert("RGB"))
                    if self.transform is not None:
                        sample = self.transform(image=sample)['image']
                    return sample, target
                except Exception as e:
                    # traceback.print_exc()
                    print(str(e), path)
                    index = random.randint(0, len(self) - 1)

        def __len__(self):
            return len(self.samples)

    ds = CustomDataset(imgs_folder, train_transforms, label_file)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf, train_transforms=None):
    if conf.data_mode == 'common':
        ds, class_num = get_custom_train_dataset(conf.train_img_dir, train_transforms)
    elif conf.data_mode == 'dacon_landmark':
        ds, class_num = get_custom_train_dataset(conf.train_img_dir, train_transforms, conf.label_file)
    else:
        if conf.data_mode in ['ms1m', 'concat']:
            ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder / 'imgs')
            print('ms1m loader generated')
        if conf.data_mode in ['vgg', 'concat']:
            vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder / 'imgs')
            print('vgg loader generated')
        if conf.data_mode == 'vgg':
            ds = vgg_ds
            class_num = vgg_class_num
        elif conf.data_mode == 'ms1m':
            ds = ms1m_ds
            class_num = ms1m_class_num
        elif conf.data_mode == 'concat':
            for i, (url, label) in enumerate(vgg_ds.imgs):
                vgg_ds.imgs[i] = (url, label + ms1m_class_num)
            ds = ConcatDataset([ms1m_ds, vgg_ds])
            class_num = vgg_class_num + ms1m_class_num
        elif conf.data_mode == 'emore':
            ds, class_num = get_train_dataset(conf.emore_folder / 'imgs')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers)
    return loader, class_num


def load_bin(path, rootdir, transform, image_size=[112, 112]):
    import mxnet as mx
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir) + '_list', np.array(issame_list))
    return data, issame_list


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=path / name, mode='r')
    issame = np.load(path / '{}_list.npy'.format(name))
    return carray, issame


def get_common_val_data(data_path, max_positive_cnt, batch_size, pin_memory, num_workers, val_transforms=None,
                        use_pos=True, use_neg=True):
    class ValDataset(Dataset):
        """__init__ and __len__ functions are the same as in TorchvisionDataset"""

        def __init__(self, files, transform=None):
            self.files = files
            self.transform = transform

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            file = self.files[index]
            im = Image.open(file).convert("RGB")
            sample = np.array(im)
            if self.transform is not None:
                sample = self.transform(image=sample)['image']
            return sample

        def __len__(self):
            return len(self.files)

    import glob
    import os

    if not use_pos and not use_neg:
        raise Exception("use_pos and use_neg all false")

    label_dirs = glob.glob(os.path.join(data_path, "*"))
    total_positive_cnt = 0
    label_files_list = []
    for label_dir in label_dirs:
        file_cnt = len(glob.glob(os.path.join(label_dir, "*")))
        total_positive_cnt += file_cnt * (file_cnt - 1) / 2
        label_files_list.append(glob.glob(os.path.join(label_dir, "*")))

    if not max_positive_cnt or max_positive_cnt > total_positive_cnt:
        max_positive_cnt = total_positive_cnt

    if max_positive_cnt < 1:
        raise Exception("max_positive_cnt is 0")

    positive_files = []
    issame = []
    if use_pos:
        each_cnt = max_positive_cnt / len(label_dirs)
        for label_idx, label_files in enumerate(label_files_list):
            cur_cnt = 0
            try:
                for i in range(0, len(label_files) - 1):
                    for j in range(i + 1, len(label_files)):
                        positive_files.append(label_files[i])
                        positive_files.append(label_files[j])
                        cur_cnt += 1
                        if cur_cnt >= each_cnt:
                            raise
            except:
                print("val positive label cnt", cur_cnt, os.path.basename(label_dirs[label_idx]))
                pass
        max_positive_cnt = len(positive_files) // 2
        issame += [True] * int(len(positive_files) / 2)
        print("val positive cnt", len(positive_files))
    negative_files = []
    if use_neg:
        total_negative_cnt = 0
        idx_map = {}
        while max_positive_cnt > total_negative_cnt:
            target_label_idx = random.randint(0, len(label_files_list) - 1)
            if len(label_files_list[target_label_idx]) < 1:
                continue
            target_item_idx = random.randint(0, len(label_files_list[target_label_idx]) - 1)
            neg_label_idx = random.randint(0, len(label_files_list) - 1)
            while target_label_idx == neg_label_idx:
                neg_label_idx = random.randint(0, len(label_files_list) - 1)

            if len(label_files_list[neg_label_idx]) < 1:
                continue
            neg_item_idx = random.randint(0, len(label_files_list[neg_label_idx]) - 1)

            if "{}_{}_{}_{}".format(target_label_idx, target_item_idx, neg_label_idx, neg_item_idx) in idx_map:
                continue

            negative_files.append(label_files_list[target_label_idx][target_item_idx])
            negative_files.append(label_files_list[neg_label_idx][neg_item_idx])
            total_negative_cnt += 1
        issame += [False] * int(len(negative_files) / 2)
        print("val negative cnt", len(negative_files))
    loader = DataLoader(ValDataset(positive_files + negative_files, val_transforms), batch_size=batch_size,
                        shuffle=False, pin_memory=pin_memory,
                        num_workers=num_workers)
    return loader, issame


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


def load_mx_rec(rec_path):
    import mxnet as mx
    save_path = rec_path / 'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path / '{}.jpg'.format(idx), quality=95)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label
