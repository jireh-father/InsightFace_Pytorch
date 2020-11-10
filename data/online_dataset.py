import sys
from torch.utils.data import Dataset
import os
import text_image_maker
import random
from text_image_param_parser import TextImageParamParser

import traceback
from data.hook_dataloader import HookDataset
import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations as al
from albumentations.augmentations import functional as F


class OnlineCharDataset(Dataset):
    def __init__(self, font_list, transform=None, paddings={"left": 5, "top": 5, "right": 5, "bottom": 5},
                 num_sample_each_class=100,
                 num_samples=None,
                 hangul_unicodes=[44036, 44039, 44040], eng_unicodes=None,
                 number_unicodes=None,
                 font_size=30,
                 skip_exception=True, use_debug=False,
                 change_font_in_error=True,
                 use_random_idx=True,
                 return_mask=False,
                 return_text=False,
                 use_padding_pixel=True):
        self.font_list = font_list
        classes = [os.path.splitext(os.path.basename(font_path))[0] for font_path in font_list]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_indices = list(range(len(classes)))
        self.classes = classes
        self.class_indices = class_indices
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.bg_list = bg_list
        self.num_sample_each_class = num_sample_each_class
        self.generator_param = TextImageParamParser(generation_params)
        self.cur_params = None
        self.cur_unicode_list = None
        self.cur_text_ing = None
        self.cur_text_ing_before = None
        self.cur_font = None

        self.min_chars = min_chars
        self.max_chars = max_chars
        self.hangul_unicodes = hangul_unicodes
        self.hangul_unicodes.sort()
        self.eng_unicodes = eng_unicodes
        self.eng_unicodes.sort()
        self.number_unicodes = number_unicodes
        self.number_unicodes.sort()
        self.hangul_prob = hangul_prob
        self.eng_prob = eng_prob
        self.num_prob = num_prob

        self.mix_prob = mix_prob
        self.min_change_char_ratio = min_change_char_ratio
        self.max_change_char_ratio = max_change_char_ratio
        self.hangul_change_uni_range = hangul_change_uni_range
        self.simple_img_prob = simple_img_prob
        self.font_size_range = font_size_range
        self.cur_font_size = None
        self.cur_text_params = None
        self.same_font_size_in_batch_prob = same_font_size_in_batch_prob

        self.same_text_in_batch_prob = same_text_in_batch_prob
        self.same_text_params_in_batch_prob = same_text_params_in_batch_prob

        self.use_img_persp_trans_prob = use_img_persp_trans_prob
        self.use_text_persp_trans_prob = use_text_persp_trans_prob
        self.skip_exception = skip_exception

        self.use_debug = use_debug
        self.input_size = input_size

        self.crop_start_ratio = 0.0
        self.use_same_random_crop_in_batch = use_same_random_crop_in_batch

        self.change_font_in_error = change_font_in_error
        self.use_random_idx = use_random_idx
        self.return_mask = return_mask
        self.return_text = return_text

        self.num_samples = num_samples

    def _create_random_text(self):
        char_len = random.randint(self.min_chars, self.max_chars)

        result_unicode_list = []
        if random.random() <= self.mix_prob:
            for i in range(char_len):
                r = random.random()
                if r <= self.hangul_prob:
                    result_unicode_list.append(random.choice(self.hangul_unicodes))
                elif r <= self.hangul_prob + self.num_prob:
                    result_unicode_list.append(random.choice(self.number_unicodes))
                elif r <= self.hangul_prob + self.num_prob + self.eng_prob:
                    result_unicode_list.append(random.choice(self.eng_unicodes))
                else:
                    raise Exception("error prob")
        else:
            r = random.random()
            if r <= self.hangul_prob:
                tmp_unicode_list = self.hangul_unicodes
            elif r <= self.hangul_prob + self.num_prob:
                tmp_unicode_list = self.number_unicodes
            elif r <= self.hangul_prob + self.num_prob + self.eng_prob:
                tmp_unicode_list = self.eng_unicodes
            else:
                raise Exception("error prob")

            for i in range(char_len):
                result_unicode_list.append(random.choice(tmp_unicode_list))
        return result_unicode_list

    def _sampling_text(self):
        self.cur_unicode_list = self._create_random_text()
        self.cur_text_ing_before = "".join([chr(c) for c in self.cur_unicode_list])
        self.cur_font_size = random.choice(self.font_size_range)
        self.cur_text_params = self._get_text_params()
        self.crop_start_ratio = random.random()

    def __before_hook__(self):
        self._sampling_text()

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    @staticmethod
    def get_random_minus_one_to_one():
        return random.random() * 2 - 1

    def _get_text_params(self):
        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}

        if random.random() <= self.simple_img_prob:
            if random.random() <= 0.6:
                char_params['bg_color'] = (255, 255, 255, 255)
                char_params['fg_color'] = (0, 0, 0, 255)
            else:
                char_params['bg_color'] = (0, 0, 0, 255)
                char_params['fg_color'] = (255, 255, 255, 255)
            char_params['use_bg_color'] = True
            # char_params['paddings'] = {"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05}
            char_params['text_gradient'] = None
            padding_params = self.generator_param.get_paddings_param()
            char_params.update(padding_params)
        else:
            bg_params = self.generator_param.get_bg_param()
            if "pos_ratio" in bg_params:
                char_params["bg_img_path"] = self._get_random_bg_img()
            padding_params = self.generator_param.get_paddings_param()
            text_params = self.generator_param.get_text_param()
            char_params.update(bg_params)
            char_params.update(padding_params)
            char_params.update(text_params)
        if random.random() <= self.use_img_persp_trans_prob:
            char_params["use_img_persp_trans"] = True
            persp_trans_params = []
            for j in range(4):
                x = OnlineFontDataset.get_random_minus_one_to_one()
                y = OnlineFontDataset.get_random_minus_one_to_one()
                persp_trans_params.append((x, y))
            char_params["img_persp_trans_params"] = persp_trans_params
            # a = 0.05 * ((random.random()) - (1 / 2))
            # b = 0.1 * ((random.random()) - (1 / 2))
            # char_params["img_persp_trans_params"] = [a, b]
        if random.random() <= self.use_text_persp_trans_prob:
            char_params["use_text_persp_trans"] = True
            persp_trans_params = []
            for j in range(4):
                x = OnlineFontDataset.get_random_minus_one_to_one()
                y = OnlineFontDataset.get_random_minus_one_to_one()
                persp_trans_params.append((x, y))
            char_params["text_persp_trans_params"] = persp_trans_params
            # a = 0.05 * ((random.random()) - (1 / 2))
            # b = 0.1 * ((random.random()) - (1 / 2))
            # char_params["text_persp_trans_params"] = [a, b]

        return char_params

    def create_text_image(self, index, etc_text_params=None):
        if random.random() <= self.same_text_in_batch_prob:
            cur_unicode_list = self.cur_unicode_list
        else:
            cur_unicode_list = self._create_random_text()
            self.cur_text_ing_before = "".join([chr(c) for c in cur_unicode_list])

        if random.random() <= self.same_font_size_in_batch_prob:
            font_size = self.cur_font_size
        else:
            font_size = random.choice(self.font_size_range)

        text = "".join([chr(c) for c in cur_unicode_list])
        self.cur_text_ing = text

        if random.random() <= self.same_text_params_in_batch_prob:
            char_params = self.cur_text_params
        else:
            char_params = self._get_text_params()

        char_params['font_size'] = font_size

        self.cur_params = char_params
        self.cur_font = self.font_list[index]
        # import uuid
        # char_params['output_path'] = "../testimage/{}_{}_{}.jpg".format(self.cur_unicode_list[0], index,
        #                                                                 str(uuid.uuid4()))
        # text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        char_params['output_path'] = None
        char_params['auto_chance_color_when_same'] = True
        char_params['raise_exception'] = False
        char_params['use_padding_pixel'] = True
        char_params['return_mask'] = self.return_mask
        # import json
        # print(json.dumps(char_params))
        # char_params = json.loads('{"color_mode": "RGB", "paddings": {"right": 0.1050272078196586, "top": 0.26090272932922254, "bottom": 0.17015320517900254, "left": 0.25837411687366774}, "text_border": null, "fg_color": [209, 235, 98, 166], "text_italic": false, "font_size": 15, "bg_img_path": "/home/irelin/resource/font_recognition/bgs/294.winterwax-500x500.jpg", "text_shadow": null, "use_img_persp_trans": true, "pos_ratio": [0.3, 0.2], "text_persp_trans_params": [-0.009976076882772873, 0.036253351174196216], "text_rotate": 14, "bg_img_width_ratio": 1.2, "text_blur": 0, "bg_img_height_ratio": 1.1, "use_bg_color": false, "bg_img_scale": 0.5601430892743575, "use_text_persp_trans": true, "use_binarize": false, "img_persp_trans_params": [-0.02296148027430462, 0.004213654695530833], "text_width_ratio": 1.0, "text_gradient": null, "output_path": null, "auto_chance_color_when_same": true, "text_height_ratio": 1.0}')
        # if self.use_debug:
        #     print(os.path.basename(self.font_list[index]), text)
        if etc_text_params:
            char_params.update(etc_text_params)
        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        if self.return_mask:
            img, mask = img

        if self.use_same_random_crop_in_batch:
            height, width = img.shape[:2]
            if width >= height:
                img = F.smallest_max_size(img, max_size=self.input_size, interpolation=cv2.INTER_LINEAR)
            else:
                img = F.longest_max_size(img, max_size=self.input_size, interpolation=cv2.INTER_LINEAR)
                pad_width = self.input_size - img.shape[:2][1]
                left = pad_width // 2
                right = pad_width - left
                img = F.pad_with_params(img, 0, 0, left, right, border_mode=cv2.BORDER_CONSTANT, value=0)

            height, width = img.shape[:2]
            if width > self.input_size:
                last_index = width - self.input_size
                start_index = int(self.crop_start_ratio * last_index)
                img = img[:, start_index:start_index + self.input_size, :]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        results = [img]

        if self.return_mask:
            results.append(mask)

        if self.return_text:
            results.append(text)

        if len(results) > 1:
            return results
        else:
            return img

    def _get_random_index(self):
        return random.choice(list(range(len(self.font_list))))

    def __getitem__(self, index):
        if self.use_random_idx:
            index = self._get_random_index()
        while True:
            try:
                # if self.use_debug:
                #     print("start font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
                img = self.create_text_image(index), index
                # if self.use_debug:
                #     print("end font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
                return img
            except Exception as e:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                if not self.skip_exception:
                    raise e
                if self.use_debug:
                    traceback.print_exc()
                    print("error font", "unicode list", self.cur_unicode_list,
                          "cur text image params", self.cur_params,
                          "cur text", self.cur_text_ing_before,
                          "changed text", self.cur_text_ing,
                          "font", self.cur_font,
                          "font_size", self.cur_font_size)

                if self.change_font_in_error:
                    tmp_index = self._get_random_index()
                    while tmp_index == index:
                        print("other font", tmp_index, index)
                        tmp_index = self._get_random_index()
                    index = tmp_index
                else:
                    print("resampling text")
                    self._sampling_text()
                # print("error font", self.font_list[class_idx])
                1

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        return len(self.font_list) * self.num_sample_each_class


class OnlineFontDataset(HookDataset):
    def __init__(self, font_list, transform=None, generation_params=None, bg_list=None,
                 num_sample_each_class=100,
                 num_samples=None,
                 min_chars=1, max_chars=3,
                 hangul_unicodes=[44036, 44039, 44040], eng_unicodes=None,
                 number_unicodes=None,
                 hangul_prob=0.4, eng_prob=0.3, num_prob=0.3, mix_prob=0.5,
                 min_change_char_ratio=0.33, max_change_char_ratio=1.0, hangul_change_uni_range=10,
                 simple_img_prob=0.5, font_size_range=None,
                 same_text_in_batch_prob=0.5, same_font_size_in_batch_prob=0.5,
                 same_text_params_in_batch_prob=0.5,
                 use_text_persp_trans_prob=0.25, use_img_persp_trans_prob=0.25,
                 skip_exception=True, use_debug=False, input_size=224, use_same_random_crop_in_batch=False,
                 change_font_in_error=True,
                 use_random_idx=True,
                 return_mask=False,
                 return_text=False):
        self.font_list = font_list
        classes = [os.path.splitext(os.path.basename(font_path))[0] for font_path in font_list]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_indices = list(range(len(classes)))
        self.classes = classes
        self.class_indices = class_indices
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.bg_list = bg_list
        self.num_sample_each_class = num_sample_each_class
        self.generator_param = TextImageParamParser(generation_params)
        self.cur_params = None
        self.cur_unicode_list = None
        self.cur_text_ing = None
        self.cur_text_ing_before = None
        self.cur_font = None

        self.min_chars = min_chars
        self.max_chars = max_chars
        self.hangul_unicodes = hangul_unicodes
        self.hangul_unicodes.sort()
        self.eng_unicodes = eng_unicodes
        self.eng_unicodes.sort()
        self.number_unicodes = number_unicodes
        self.number_unicodes.sort()
        self.hangul_prob = hangul_prob
        self.eng_prob = eng_prob
        self.num_prob = num_prob

        self.mix_prob = mix_prob
        self.min_change_char_ratio = min_change_char_ratio
        self.max_change_char_ratio = max_change_char_ratio
        self.hangul_change_uni_range = hangul_change_uni_range
        self.simple_img_prob = simple_img_prob
        self.font_size_range = font_size_range
        self.cur_font_size = None
        self.cur_text_params = None
        self.same_font_size_in_batch_prob = same_font_size_in_batch_prob

        self.same_text_in_batch_prob = same_text_in_batch_prob
        self.same_text_params_in_batch_prob = same_text_params_in_batch_prob

        self.use_img_persp_trans_prob = use_img_persp_trans_prob
        self.use_text_persp_trans_prob = use_text_persp_trans_prob
        self.skip_exception = skip_exception

        self.use_debug = use_debug
        self.input_size = input_size

        self.crop_start_ratio = 0.0
        self.use_same_random_crop_in_batch = use_same_random_crop_in_batch

        self.change_font_in_error = change_font_in_error
        self.use_random_idx = use_random_idx
        self.return_mask = return_mask
        self.return_text = return_text

        self.num_samples = num_samples

    def _create_random_text(self):
        char_len = random.randint(self.min_chars, self.max_chars)

        result_unicode_list = []
        if random.random() <= self.mix_prob:
            for i in range(char_len):
                r = random.random()
                if r <= self.hangul_prob:
                    result_unicode_list.append(random.choice(self.hangul_unicodes))
                elif r <= self.hangul_prob + self.num_prob:
                    result_unicode_list.append(random.choice(self.number_unicodes))
                elif r <= self.hangul_prob + self.num_prob + self.eng_prob:
                    result_unicode_list.append(random.choice(self.eng_unicodes))
                else:
                    raise Exception("error prob")
        else:
            r = random.random()
            if r <= self.hangul_prob:
                tmp_unicode_list = self.hangul_unicodes
            elif r <= self.hangul_prob + self.num_prob:
                tmp_unicode_list = self.number_unicodes
            elif r <= self.hangul_prob + self.num_prob + self.eng_prob:
                tmp_unicode_list = self.eng_unicodes
            else:
                raise Exception("error prob")

            for i in range(char_len):
                result_unicode_list.append(random.choice(tmp_unicode_list))
        return result_unicode_list

    def _sampling_text(self):
        self.cur_unicode_list = self._create_random_text()
        self.cur_text_ing_before = "".join([chr(c) for c in self.cur_unicode_list])
        self.cur_font_size = random.choice(self.font_size_range)
        self.cur_text_params = self._get_text_params()
        self.crop_start_ratio = random.random()

    def __before_hook__(self):
        self._sampling_text()

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    @staticmethod
    def get_random_minus_one_to_one():
        return random.random() * 2 - 1

    def _get_text_params(self):
        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}

        if random.random() <= self.simple_img_prob:
            if random.random() <= 0.6:
                char_params['bg_color'] = (255, 255, 255, 255)
                char_params['fg_color'] = (0, 0, 0, 255)
            else:
                char_params['bg_color'] = (0, 0, 0, 255)
                char_params['fg_color'] = (255, 255, 255, 255)
            char_params['use_bg_color'] = True
            # char_params['paddings'] = {"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05}
            char_params['text_gradient'] = None
            padding_params = self.generator_param.get_paddings_param()
            char_params.update(padding_params)
        else:
            bg_params = self.generator_param.get_bg_param()
            if "pos_ratio" in bg_params:
                char_params["bg_img_path"] = self._get_random_bg_img()
            padding_params = self.generator_param.get_paddings_param()
            text_params = self.generator_param.get_text_param()
            char_params.update(bg_params)
            char_params.update(padding_params)
            char_params.update(text_params)
        if random.random() <= self.use_img_persp_trans_prob:
            char_params["use_img_persp_trans"] = True
            persp_trans_params = []
            for j in range(4):
                x = OnlineFontDataset.get_random_minus_one_to_one()
                y = OnlineFontDataset.get_random_minus_one_to_one()
                persp_trans_params.append((x, y))
            char_params["img_persp_trans_params"] = persp_trans_params
            # a = 0.05 * ((random.random()) - (1 / 2))
            # b = 0.1 * ((random.random()) - (1 / 2))
            # char_params["img_persp_trans_params"] = [a, b]
        if random.random() <= self.use_text_persp_trans_prob:
            char_params["use_text_persp_trans"] = True
            persp_trans_params = []
            for j in range(4):
                x = OnlineFontDataset.get_random_minus_one_to_one()
                y = OnlineFontDataset.get_random_minus_one_to_one()
                persp_trans_params.append((x, y))
            char_params["text_persp_trans_params"] = persp_trans_params
            # a = 0.05 * ((random.random()) - (1 / 2))
            # b = 0.1 * ((random.random()) - (1 / 2))
            # char_params["text_persp_trans_params"] = [a, b]

        return char_params

    def create_text_image(self, index, etc_text_params=None):
        if random.random() <= self.same_text_in_batch_prob:
            cur_unicode_list = self.cur_unicode_list
        else:
            cur_unicode_list = self._create_random_text()
            self.cur_text_ing_before = "".join([chr(c) for c in cur_unicode_list])

        if random.random() <= self.same_font_size_in_batch_prob:
            font_size = self.cur_font_size
        else:
            font_size = random.choice(self.font_size_range)

        text = "".join([chr(c) for c in cur_unicode_list])
        self.cur_text_ing = text

        if random.random() <= self.same_text_params_in_batch_prob:
            char_params = self.cur_text_params
        else:
            char_params = self._get_text_params()

        char_params['font_size'] = font_size

        self.cur_params = char_params
        self.cur_font = self.font_list[index]
        # import uuid
        # char_params['output_path'] = "../testimage/{}_{}_{}.jpg".format(self.cur_unicode_list[0], index,
        #                                                                 str(uuid.uuid4()))
        # text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        char_params['output_path'] = None
        char_params['auto_chance_color_when_same'] = True
        char_params['raise_exception'] = False
        char_params['return_mask'] = self.return_mask
        # import json
        # print(json.dumps(char_params))
        # char_params = json.loads('{"color_mode": "RGB", "paddings": {"right": 0.1050272078196586, "top": 0.26090272932922254, "bottom": 0.17015320517900254, "left": 0.25837411687366774}, "text_border": null, "fg_color": [209, 235, 98, 166], "text_italic": false, "font_size": 15, "bg_img_path": "/home/irelin/resource/font_recognition/bgs/294.winterwax-500x500.jpg", "text_shadow": null, "use_img_persp_trans": true, "pos_ratio": [0.3, 0.2], "text_persp_trans_params": [-0.009976076882772873, 0.036253351174196216], "text_rotate": 14, "bg_img_width_ratio": 1.2, "text_blur": 0, "bg_img_height_ratio": 1.1, "use_bg_color": false, "bg_img_scale": 0.5601430892743575, "use_text_persp_trans": true, "use_binarize": false, "img_persp_trans_params": [-0.02296148027430462, 0.004213654695530833], "text_width_ratio": 1.0, "text_gradient": null, "output_path": null, "auto_chance_color_when_same": true, "text_height_ratio": 1.0}')
        # if self.use_debug:
        #     print(os.path.basename(self.font_list[index]), text)
        if etc_text_params:
            char_params.update(etc_text_params)
        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        if self.return_mask:
            img, mask = img

        if self.use_same_random_crop_in_batch:
            height, width = img.shape[:2]
            if width >= height:
                img = F.smallest_max_size(img, max_size=self.input_size, interpolation=cv2.INTER_LINEAR)
            else:
                img = F.longest_max_size(img, max_size=self.input_size, interpolation=cv2.INTER_LINEAR)
                pad_width = self.input_size - img.shape[:2][1]
                left = pad_width // 2
                right = pad_width - left
                img = F.pad_with_params(img, 0, 0, left, right, border_mode=cv2.BORDER_CONSTANT, value=0)

            height, width = img.shape[:2]
            if width > self.input_size:
                last_index = width - self.input_size
                start_index = int(self.crop_start_ratio * last_index)
                img = img[:, start_index:start_index + self.input_size, :]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        results = [img]

        if self.return_mask:
            results.append(mask)

        if self.return_text:
            results.append(text)

        if len(results) > 1:
            return results
        else:
            return img

    def _get_random_index(self):
        return random.choice(list(range(len(self.font_list))))

    def __getitem__(self, index):
        if self.use_random_idx:
            index = self._get_random_index()
        while True:
            try:
                # if self.use_debug:
                #     print("start font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
                img = self.create_text_image(index), index
                # if self.use_debug:
                #     print("end font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
                return img
            except Exception as e:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                if not self.skip_exception:
                    raise e
                if self.use_debug:
                    traceback.print_exc()
                    print("error font", "unicode list", self.cur_unicode_list,
                          "cur text image params", self.cur_params,
                          "cur text", self.cur_text_ing_before,
                          "changed text", self.cur_text_ing,
                          "font", self.cur_font,
                          "font_size", self.cur_font_size)

                if self.change_font_in_error:
                    tmp_index = self._get_random_index()
                    while tmp_index == index:
                        print("other font", tmp_index, index)
                        tmp_index = self._get_random_index()
                    index = tmp_index
                else:
                    print("resampling text")
                    self._sampling_text()
                # print("error font", self.font_list[class_idx])
                1

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        return len(self.font_list) * self.num_sample_each_class
