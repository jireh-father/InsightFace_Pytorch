import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as al
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2


def get_common_train_transforms():
    return [
        # al.OneOf([
        #     al.RandomRotate90(),
        #     al.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT),
        # ], p=0.1),
        al.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=30, p=0.05),
        al.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=5.0, shift_limit=0.1, p=0.05),
        al.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=15, p=0.05),
        al.OneOf([
            al.RandomGamma(),
            al.HueSaturationValue(),
            al.RGBShift(),
            al.CLAHE(),
            al.ChannelShuffle(),
            al.InvertImg(),
        ], p=0.1),

        al.RandomSnow(p=0.05),
        al.RandomRain(p=0.05),
        # al.RandomFog(p=0.05),
        # al.RandomSunFlare(p=0.05),
        al.RandomShadow(p=0.05),
        al.RandomBrightnessContrast(p=0.2),
        al.GaussNoise(p=0.2),
        al.ISONoise(p=0.05),
        al.ToGray(p=0.05),
        al.OneOf([
            # al.MotionBlur(blur_limit=4),
            al.Blur(blur_limit=2),
            # al.MedianBlur(blur_limit=4),
            # al.GaussianBlur(blur_limit=4),
        ], p=0.05),
        al.CoarseDropout(p=0.05),
        al.Downscale(p=0.05),
        al.ImageCompression(quality_lower=60, p=0.2),
    ]


def _get_train_pixel_transforms(use_gray=False, use_blur=False):
    return [
        al.RandomGamma(p=0.05),
        al.HueSaturationValue(p=0.05),
        al.RGBShift(p=0.05),
        al.CLAHE(p=0.05),
        al.ChannelShuffle(p=0.05),
        al.InvertImg(p=0.05),

        al.RandomSnow(p=0.05),
        al.RandomRain(p=0.05),
        al.RandomFog(p=0.05),
        al.RandomSunFlare(p=0.05, num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110),
        al.RandomShadow(p=0.05),
        al.RandomBrightnessContrast(p=0.05),
        al.GaussNoise(p=0.05),
        al.ISONoise(p=0.05),
        al.MultiplicativeNoise(p=0.05),
        al.ToGray(p=1.0 if use_gray else 0.05),
        al.ToSepia(p=0.05),
        al.Solarize(p=0.05),
        al.Equalize(p=0.05),
        al.Posterize(p=0.05),
        al.FancyPCA(p=0.05),
        al.OneOf([
            al.MotionBlur(blur_limit=1),
            al.Blur(blur_limit=1),
            al.MedianBlur(blur_limit=1),
            al.GaussianBlur(blur_limit=1),
        ], p=0.05 if use_blur else 0.),
        al.CoarseDropout(p=0.05),
        al.Cutout(p=0.05),
        al.GridDropout(p=0.05),
        al.ChannelDropout(p=0.05),
        al.Downscale(p=0.1),
        al.ImageCompression(quality_lower=60, p=0.1),
    ]


def _get_train_transforms(use_gray=False, only_use_pixel_transform=False, use_flip=False, use_blur=False):
    pixel_transforms = _get_train_pixel_transforms(use_gray, use_blur)
    if only_use_pixel_transform:
        return pixel_transforms
    else:
        return [
                   al.Flip(p=0.5 if use_flip else 0.),
                   al.OneOf([
                       al.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT),
                   ], p=0.05),
                   al.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=30, p=0.05),
                   al.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=5.0, shift_limit=0.1, p=0.05),
                   al.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.05),
                   al.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=15, p=0.05),

                   al.RandomGridShuffle(p=0.05),
               ] + pixel_transforms


def get_train_transforms(input_size=None, use_random_crop=False, use_gray=False, use_online=True,
                         use_same_random_crop_in_batch=False, use_normalize=True, only_use_pixel_transform=False,
                         use_flip=False, use_blur=False):
    if not use_online:
        return al.Compose(_get_train_transforms(use_gray, only_use_pixel_transform, use_flip, use_blur))

    def resize_image(img, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        if width >= height:
            img = F.smallest_max_size(img, max_size=input_size, interpolation=interpolation)
        else:
            img = F.longest_max_size(img, max_size=input_size, interpolation=interpolation)
            pad_width = input_size - img.shape[:2][1]
            left = pad_width // 2
            right = pad_width - left
            img = F.pad_with_params(img, 0, 0, left, right, border_mode=cv2.BORDER_CONSTANT, value=0)

        return img

    def left_crop(img, **params):
        height, width = img.shape[:2]
        if width > input_size:
            img = img[:, :input_size, :]
        return img

    if use_same_random_crop_in_batch:
        compose = _get_train_transforms(use_gray, only_use_pixel_transform, use_flip, use_blur)
    else:
        if use_random_crop:
            crop = al.RandomCrop(input_size, input_size)
        else:
            crop = al.Lambda(left_crop)

        compose = [  # al.SmallestMaxSize(input_size),
                      al.Lambda(resize_image),
                      crop] + _get_train_transforms(use_gray, only_use_pixel_transform, use_flip, use_blur)
    if use_normalize:
        return al.Compose(compose +
                          [
                              al.Normalize(),
                              ToTensorV2()
                          ])

    else:
        return al.Compose(compose)


def get_simple_transforms(input_size=224, use_random_crop=False, use_same_random_crop_in_batch=False, use_gray=False):
    def resize_image(img, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        if width >= height:
            img = F.smallest_max_size(img, max_size=input_size, interpolation=interpolation)
        else:
            img = F.longest_max_size(img, max_size=input_size, interpolation=interpolation)
            pad_width = input_size - img.shape[:2][1]
            left = pad_width // 2
            right = pad_width - left
            img = F.pad_with_params(img, 0, 0, left, right, border_mode=cv2.BORDER_CONSTANT, value=0)

        return img

    def left_crop(img, **params):
        height, width = img.shape[:2]
        if width > input_size:
            img = img[:, :input_size, :]
        return img

    if use_same_random_crop_in_batch:
        compose = []
    else:
        if use_random_crop:
            crop = al.RandomCrop(input_size, input_size)
        else:
            crop = al.Lambda(left_crop)

        compose = [  # al.SmallestMaxSize(input_size),
            al.Lambda(resize_image),
            crop]
    if use_gray:
        return al.Compose(compose +
                          [
                              al.ToGray(p=1.0 if use_gray else 0.05),
                              al.Normalize(),
                              ToTensorV2()
                          ])

    else:
        return al.Compose(compose +
                          [
                              al.Normalize(),
                              ToTensorV2()
                          ])


def get_val_transforms(input_size=224, use_gray=False):
    def resize_image(img, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        if width >= height:
            img = F.smallest_max_size(img, max_size=input_size, interpolation=interpolation)
        else:
            img = F.longest_max_size(img, max_size=input_size, interpolation=interpolation)
            pad_width = input_size - img.shape[:2][1]
            left = pad_width // 2
            right = pad_width - left
            img = F.pad_with_params(img, 0, 0, left, right, border_mode=cv2.BORDER_CONSTANT, value=0)

        return img

    def left_crop(img, **params):
        height, width = img.shape[:2]
        if width > input_size:
            img = img[:, :input_size, :]
        return img

    crop = al.Lambda(left_crop)

    compose = [  # al.SmallestMaxSize(input_size),
        al.Lambda(resize_image),
        crop]
    if use_gray:
        return al.Compose(compose +
                          [
                              al.ToGray(p=1.0 if use_gray else 0.05),
                              al.Normalize(),
                              ToTensorV2()
                          ])

    else:
        return al.Compose(compose +
                          [
                              al.Normalize(),
                              ToTensorV2()
                          ])
