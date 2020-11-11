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


def get_train_common_transforms(input_size=None, use_random_crop=False, use_gray=False, only_use_pixel_transform=False,
                                use_flip=False, use_blur=False, no_transforms=False,
                                use_center_crop=False,
                                center_crop_ratio=0.8):
    if use_random_crop:
        compose = [al.Resize(int(input_size * 1.1), int(input_size * 1.1)),
                   al.RandomCrop(input_size, input_size)]
    elif use_center_crop:
        compose = [al.Resize(int(input_size * (2.0 - center_crop_ratio)), int(input_size * (2.0 - center_crop_ratio))),
                   al.CenterCrop(input_size, input_size)]
    else:
        compose = [al.Resize(input_size, input_size)]

    if no_transforms:
        return al.Compose(compose +
                          [
                              al.Normalize(),
                              ToTensorV2()
                          ])

    return al.Compose(compose + _get_train_transforms(use_gray, only_use_pixel_transform, use_flip, use_blur) +
                      [
                          al.Normalize(),
                          ToTensorV2()
                      ])


def get_train_transforms_simple(input_size=None, use_random_crop=False, use_gray=False, only_use_pixel_transform=False,
                                use_flip=False, use_blur=False, no_transforms=False,
                                use_center_crop=False,
                                center_crop_ratio=0.8):
    return al.Compose(
        [
            al.Resize(input_size, input_size, p=1.0),
            al.HorizontalFlip(p=0.5),
            al.Normalize(),
            ToTensorV2()
        ])


def get_train_transforms_mmdetection(input_size=None, use_random_crop=False, use_gray=False,
                                     only_use_pixel_transform=False,
                                     use_flip=False, use_blur=False, no_transforms=False,
                                     use_center_crop=False,
                                     center_crop_ratio=0.8):
    return al.Compose(
        [
            al.RandomResizedCrop(height=input_size,
                                 width=input_size,
                                 scale=(0.4, 1.0),
                                 interpolation=0,
                                 p=0.5),
            al.Resize(input_size, input_size, p=1.0),
            al.HorizontalFlip(p=0.5),
            al.OneOf([
                al.ShiftScaleRotate(border_mode=0,
                                    shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2),
                                    rotate_limit=(-20, 20)),
                al.OpticalDistortion(border_mode=0,
                                     distort_limit=[-0.5, 0.5], shift_limit=[-0.5, 0.5]),
                al.GridDistortion(num_steps=5, distort_limit=[-0., 0.3], border_mode=0),
                al.ElasticTransform(border_mode=0),
                al.IAAPerspective(),
                al.RandomGridShuffle()
            ], p=0.1),
            al.Rotate(limit=(-25, 25), border_mode=0, p=0.1),
            al.OneOf([
                al.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2)),
                al.HueSaturationValue(hue_shift_limit=(-20, 20),
                                      sat_shift_limit=(-30, 30),
                                      val_shift_limit=(-20, 20)),
                al.RandomGamma(gamma_limit=(30, 150)),
                al.RGBShift(),
                al.CLAHE(clip_limit=(1, 15)),
                al.ChannelShuffle(),
                al.InvertImg(),
            ], p=0.1),
            al.RandomSnow(p=0.05),
            al.RandomRain(p=0.05),
            al.RandomFog(p=0.05),
            al.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110, p=0.05),
            al.RandomShadow(p=0.05),

            al.GaussNoise(var_limit=(10, 20), p=0.05),
            al.ISONoise(color_shift=(0, 15), p=0.05),
            al.MultiplicativeNoise(p=0.05),
            al.OneOf([
                al.ToGray(p=1. if use_gray else 0.05),
                al.ToSepia(p=0.05),
                al.Solarize(p=0.05),
                al.Equalize(p=0.05),
                al.Posterize(p=0.05),
                al.FancyPCA(p=0.05),
            ], p=0.05),
            al.OneOf([
                al.MotionBlur(blur_limit=(3, 7)),
                al.Blur(blur_limit=(3, 7)),
                al.MedianBlur(blur_limit=3),
                al.GaussianBlur(blur_limit=3),
            ], p=0.05),
            al.CoarseDropout(p=0.05),
            al.Cutout(num_holes=30, max_h_size=37, max_w_size=37, fill_value=0, p=0.05),
            al.GridDropout(p=0.05),
            al.ChannelDropout(p=0.05),
            al.Downscale(scale_min=0.5, scale_max=0.9, p=0.1),
            al.ImageCompression(quality_lower=60, p=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_val_common_transforms(input_size=None, use_random_crop=False, use_gray=False,
                              use_center_crop=False,
                              center_crop_ratio=0.8):
    if use_random_crop:
        compose = [al.Resize(int(input_size * 1.1), int(input_size * 1.1)),
                   al.CenterCrop(input_size, input_size)]
    elif use_center_crop:
        compose = [al.Resize(int(input_size * (2.0 - center_crop_ratio)), int(input_size * (2.0 - center_crop_ratio))),
                   al.CenterCrop(input_size, input_size)]
    else:
        compose = [al.Resize(input_size, input_size)]

    return al.Compose(compose + [
        al.ToGray(p=1.0 if use_gray else 0.0),
        al.Normalize(),
        ToTensorV2()
    ])


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


def get_test_transforms(input_size, use_gray=False):
    # return transforms.Compose([
    #     transforms.Resize(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    return al.Compose([
        al.SmallestMaxSize(input_size),
        al.ToGray(p=1.0 if use_gray else 0.0),
        al.Normalize(),
        ToTensorV2()
    ])

def get_test_transforms_v2(input_size, use_crop=False, center_crop_ratio=0.9, use_gray=False):
    if use_crop:
        resize = [al.Resize(int(input_size * (2 - center_crop_ratio)),
                            int(input_size * (2 - center_crop_ratio))),
                  al.CenterCrop(height=input_size, width=input_size)]
    else:
        resize = [al.Resize(input_size, input_size)]
    return al.Compose(resize + [
        al.ToGray(p=1. if use_gray else 0.),
        al.Normalize(),
        ToTensorV2()
    ])
