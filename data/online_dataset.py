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


class SimpleOnlineDataset(Dataset):
    def __init__(self, font_list, font_size=185, transform=None, text=None):
        self.font_list = font_list

        classes = [os.path.splitext(os.path.basename(font_path))[0] for font_path in font_list]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_indices = list(range(len(classes)))
        self.classes = classes
        self.class_indices = class_indices
        self.class_to_idx = class_to_idx
        self.len = sys.maxsize
        self.font_size = font_size
        self.transform = transform
        self.text = text

    def create_text_image(self, index):
        img = text_image_maker.create_text_image(self.text, font_path=self.font_list[index], font_size=self.font_size)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, index

    def __getitem__(self, index):
        while True:
            try:
                return self.create_text_image(index)
            except Exception:
                traceback.print_exc()
                print("error font", self.font_list[index])

    def __len__(self):
        if self.text is None:
            return len(self.font_list) * 1000
        else:
            return len(self.font_list)


class OnlineRandomDataset(Dataset):
    def __init__(self, font_list, char_unicode_dict=None, transform=None,
                 fixed_char_unicode=None, generation_params=None, bg_list=None, num_sample_each_class=1000,
                 min_num_chars=1, max_num_chars=10):
        self.font_list = font_list

        classes = [os.path.splitext(os.path.basename(font_path))[0] for font_path in font_list]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_indices = list(range(len(classes)))
        self.classes = classes
        self.class_indices = class_indices
        self.class_to_idx = class_to_idx
        self.char_unicode_dict = char_unicode_dict
        self.transform = transform
        self.fixed_char_unicode = fixed_char_unicode
        self.bg_list = bg_list
        self.num_sample_each_class = num_sample_each_class
        self.generator_param = TextImageParamParser(generation_params)
        self.min_num_chars = min_num_chars
        self.max_num_chars = max_num_chars
        self.cur_params = None
        self.cur_text = None
        self.cur_font = None

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    def create_text_image(self, index):
        if self.fixed_char_unicode is None:
            num_chars = random.randint(self.min_num_chars, self.max_num_chars)
            char_list = []
            for i in range(num_chars):
                prob = random.random()
                acc_prob = 0.
                char_unicode_list = None
                for key in self.char_unicode_dict:
                    acc_prob += self.char_unicode_dict[key]['prob']
                    if acc_prob >= prob:
                        char_unicode_list = self.char_unicode_dict[key]['unicode']
                        break
                if char_unicode_list is None:
                    raise Exception("unicode list의 확률값이 이상합니다.")

                char_list.append(chr(random.choice(char_unicode_list)))
            text = "".join(char_list)
        else:
            text = chr(self.fixed_char_unicode)
        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}

        bg_params = self.generator_param.get_bg_param()
        if "pos_ratio" in bg_params:
            char_params["bg_img_path"] = self._get_random_bg_img()
        padding_params = self.generator_param.get_paddings_param()
        text_params = self.generator_param.get_text_param()
        char_params.update(bg_params)
        char_params.update(padding_params)
        char_params.update(text_params)
        # print(char_params)
        self.cur_params = char_params
        self.cur_text = text
        self.cur_font = self.font_list[index]
        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img

    def __getitem__(self, index):
        while True:
            try:
                class_idx = random.randint(0, len(self.classes) - 1)
                img = self.create_text_image(class_idx), class_idx
                # print("success")
                return img
            except OSError as e:
                if isinstance(e.args[0], str) and e.args[0].startswith("cannot identify image file"):
                    if os.path.isfile(self.cur_params["bg_img_path"]):
                        os.unlink(self.cur_params["bg_img_path"])
            except Exception:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                # traceback.print_exc()
                # print("error font", self.font_list[class_idx])
                1

    def __len__(self):
        if self.fixed_char_unicode is None:
            return len(self.font_list) * self.num_sample_each_class
        else:
            return len(self.font_list)


class MatchOnlineDiffTextDataset(Dataset):
    def __init__(self, font_list, transform=None, generation_params=None, bg_list=None, num_sample_each_class=100,
                 min_chars=1, max_chars=3,
                 hangul_unicodes=None, eng_unicodes=None,
                 number_unicodes=None,
                 hangul_prob=0.4, eng_prob=0.3, num_prob=0.3, mix_prob=0.5, change_prob=0.25,
                 simple_img_prob=0.5, font_size_range=None):
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
        self.change_prob = change_prob
        self.simple_img_prob = simple_img_prob
        self.font_size_range = font_size_range
        self.cur_font_size = None

    def _sampling_text(self):
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
        self.cur_unicode_list = result_unicode_list
        self.cur_font_size = random.choice(self.font_size_range)

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    def create_text_image(self, index):
        self._sampling_text()

        text = "".join([chr(c) for c in self.cur_unicode_list])

        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}

        if random.random() <= self.simple_img_prob:
            char_params['bg_color'] = (255, 255, 255, 255)
            char_params['use_bg_color'] = True
            char_params['paddings'] = {"left": 2, "top": 2, "right": 2, "bottom": 2}
            char_params['fg_color'] = (0, 0, 0, 255)
            char_params['text_gradient'] = None
        else:
            bg_params = self.generator_param.get_bg_param()
            if "pos_ratio" in bg_params:
                char_params["bg_img_path"] = self._get_random_bg_img()
            padding_params = self.generator_param.get_paddings_param()
            text_params = self.generator_param.get_text_param()
            char_params.update(bg_params)
            char_params.update(padding_params)
            char_params.update(text_params)

        char_params['font_size'] = self.cur_font_size + random.randint(0, 15)

        self.cur_params = char_params
        self.cur_font = self.font_list[index]
        # import uuid
        # char_params['output_path'] = "../testimage/{}_{}_{}.jpg".format(self.cur_unicode_list[0], index,
        #                                                                 str(uuid.uuid4()))
        # text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        char_params['output_path'] = None

        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img

    def __getitem__(self, index):
        while True:
            try:
                img = self.create_text_image(index), index
                return img
            except Exception:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                # traceback.print_exc()
                # print("error font", self.font_list[class_idx])
                1

    def __len__(self):
        return len(self.font_list) * self.num_sample_each_class


class MatchOnlineRandomDataset(HookDataset):
    CHOSUNG_SIM_LIST = {
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄲ': ['ㄱ'],
        'ㄴ': ['ㄴ', 'ㄷ'],
        'ㄷ': ['ㄸ', 'ㅁ', 'ㅌ'],
        'ㄸ': ['ㄷ', 'ㅁ', 'ㅌ'],
        'ㄹ': ['ㄷ'],
        'ㅁ': ['ㅇ', 'ㅍ', 'ㅂ'],
        'ㅂ': ['ㅁ'],
        'ㅃ': ['ㅂ', 'ㅁ'],
        'ㅅ': ['ㅆ', 'ㅈ'],
        'ㅆ': ['ㅅ', 'ㅉ'],
        'ㅇ': ['ㅁ', 'ㅎ'],
        'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ'],
        'ㅉ': ['ㅆ', 'ㅈ'],
        'ㅊ': ['ㅉ', 'ㅈ'],
        'ㅋ': ['ㄱ', 'ㄲ'],
        'ㅌ': ['ㄷ'],
        'ㅍ': ['ㅁ', 'ㅂ'],
        'ㅎ': ['ㅇ']
    }
    JUNGSUNG_SIM_LIST = {
        'ㅏ': ['ㅑ', 'ㅣ'],
        'ㅐ': ['ㅣ', 'ㅒ', 'ㅔ'],
        'ㅑ': ['ㅏ', 'ㅣ'],
        'ㅒ': ['ㅐ', 'ㅖ'],
        'ㅓ': ['ㅕ', 'ㅣ', 'ㅔ'],
        'ㅔ': ['ㅓ', 'ㅖ', 'ㅐ'],
        'ㅕ': ['ㅓ', 'ㅣ'],
        'ㅖ': ['ㅕ', 'ㅐ'],
        'ㅗ': ['ㅡ', 'ㅛ'],
        'ㅘ': ['ㅚ'],
        'ㅙ': ['ㅚ'],
        'ㅚ': ['ㅘ', 'ㅗㅒ'],
        'ㅛ': ['ㅗ'],
        'ㅜ': ['ㅠ', 'ㅡ'],
        'ㅝ': ['ㅞ', 'ㅟ'],
        'ㅞ': ['ㅝ'],
        'ㅟ': ['ㅝ'],
        'ㅠ': ['ㅜ'],
        'ㅡ': ['ㅜ', 'ㅗ'],
        'ㅢ': ['ㅝ'],
        'ㅣ': ['ㅏ', 'ㅓ'],
    }

    JONGSUNG_SIM_LIST = {
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄲ': ['ㄱ'],
        'ㄳ': ['ㄲ'],
        'ㄴ': ['ㄴ', 'ㄷ'],
        'ㄵ': ['ㄶ'],
        'ㄶ': ['ㄵ'],
        'ㄷ': ['ㄸ', 'ㅁ', 'ㅌ'],
        'ㄹ': ['ㄷ'],
        'ㄺ': ['ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄻ': ['ㄺ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄼ': ['ㄺ', 'ㄻ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄽ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄿ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㅀ'],
        'ㄾ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄿ', 'ㅀ'],
        'ㅀ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ'],
        'ㅁ': ['ㅇ', 'ㅍ', 'ㅂ'],
        'ㅂ': ['ㅁ'],
        'ㅄ': ['ㄽ'],
        'ㅅ': ['ㅆ', 'ㅈ'],
        'ㅆ': ['ㅅ', 'ㅉ'],
        'ㅇ': ['ㅁ', 'ㅎ'],
        'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ'],
        'ㅊ': ['ㅉ', 'ㅈ'],
        'ㅋ': ['ㄱ', 'ㄲ'],
        'ㅌ': ['ㄷ'],
        'ㅍ': ['ㅁ', 'ㅂ'],
        'ㅎ': ['ㅇ']
    }

    def __init__(self, font_list, transform=None, generation_params=None, bg_list=None, num_sample_each_class=1000,
                 min_chars=1, max_chars=3,
                 hangul_unicodes=[44036, 44039, 44040], eng_unicodes=None,
                 number_unicodes=None,
                 hangul_prob=0.4, eng_prob=0.3, num_prob=0.3, mix_prob=0.5, change_prob=0.25,
                 min_change_char_ratio=0.33, max_change_char_ratio=1.0, hangul_change_uni_range=10,
                 simple_img_prob=0.5, font_size_range=None):
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
        self.change_prob = change_prob
        self.min_change_char_ratio = min_change_char_ratio
        self.max_change_char_ratio = max_change_char_ratio
        self.hangul_change_uni_range = hangul_change_uni_range
        self.simple_img_prob = simple_img_prob
        self.font_size_range = font_size_range
        self.cur_font_size = None

    def _sampling_text(self):
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
        self.cur_unicode_list = result_unicode_list
        self.cur_text_ing_before = "".join([chr(c) for c in self.cur_unicode_list])
        self.cur_font_size = random.choice(self.font_size_range)

    def __before_hook__(self):
        self._sampling_text()

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    def create_text_image(self, index):
        import hgtk
        cur_unicode_list = []

        if random.random() <= self.change_prob:
            change_char_ratio = random.uniform(self.min_change_char_ratio, self.max_change_char_ratio)
            num_change = int(len(self.cur_unicode_list) * change_char_ratio)
            change_idx_list = list(range(len(self.cur_unicode_list)))
            random.shuffle(change_idx_list)
            if num_change < 1:
                num_change = 1
            change_idx_list = change_idx_list[:num_change]

            for i, uni_num in enumerate(self.cur_unicode_list):
                if i in change_idx_list:
                    if self.hangul_unicodes[0] <= uni_num <= self.hangul_unicodes[-1]:
                        if random.random() <= 0.5:
                            dec = hgtk.letter.decompose(chr(uni_num))
                            chosung = dec[0]
                            jungsung = dec[1]
                            jongsung = dec[2]
                            chosung = random.choice(MatchOnlineRandomDataset.CHOSUNG_SIM_LIST[chosung])
                            jungsung = random.choice(MatchOnlineRandomDataset.JUNGSUNG_SIM_LIST[jungsung])

                            if jongsung == '':
                                if random.random() > 0.8:
                                    k = random.choice(list(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST.keys()))
                                    jongsung = random.choice(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST[k])
                            else:
                                if random.random() > 0.8:
                                    jongsung = ''
                                else:
                                    jongsung = random.choice(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST[jongsung])

                            tmp_uni_num = ord(hgtk.letter.compose(chosung, jungsung, jongsung))
                            if tmp_uni_num not in self.hangul_unicodes:
                                cur_unicode_list.append(uni_num)
                            else:
                                cur_unicode_list.append(tmp_uni_num)
                        else:
                            uni_idx = self.hangul_unicodes.index(uni_num)
                            min_idx = uni_idx - self.hangul_change_uni_range
                            max_idx = uni_idx + self.hangul_change_uni_range
                            if min_idx < 0:
                                min_idx = 0
                            if max_idx > len(self.hangul_unicodes):
                                max_idx = len(self.hangul_unicodes)

                            cur_unicode_list.append(self.hangul_unicodes[random.randrange(min_idx, max_idx)])

                    elif self.number_unicodes[0] <= uni_num <= self.number_unicodes[-1]:
                        cur_unicode_list.append(random.choice(self.number_unicodes))
                    elif self.eng_unicodes[0] <= uni_num <= self.eng_unicodes[-1]:
                        cur_unicode_list.append(random.choice(self.eng_unicodes))
                    else:
                        raise Exception("error unicode range")
                else:
                    cur_unicode_list.append(uni_num)
        else:
            cur_unicode_list = self.cur_unicode_list

        text = "".join([chr(c) for c in cur_unicode_list])
        self.cur_text_ing = text

        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}

        if random.random() <= self.simple_img_prob:
            char_params['bg_color'] = (255, 255, 255, 255)
            char_params['use_bg_color'] = True
            char_params['paddings'] = {"left": 2, "top": 2, "right": 2, "bottom": 2}
            char_params['fg_color'] = (0, 0, 0, 255)
            char_params['text_gradient'] = None
        else:
            bg_params = self.generator_param.get_bg_param()
            if "pos_ratio" in bg_params:
                char_params["bg_img_path"] = self._get_random_bg_img()
            padding_params = self.generator_param.get_paddings_param()
            text_params = self.generator_param.get_text_param()
            char_params.update(bg_params)
            char_params.update(padding_params)
            char_params.update(text_params)

        char_params['font_size'] = self.cur_font_size + random.randint(0, 15)

        self.cur_params = char_params
        self.cur_font = self.font_list[index]
        # import uuid
        # char_params['output_path'] = "../testimage/{}_{}_{}.jpg".format(self.cur_unicode_list[0], index,
        #                                                                 str(uuid.uuid4()))
        # text_image_maker.create_text_image(text, self.font_list[index], **char_params)
        char_params['output_path'] = None
        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img

    def __getitem__(self, index):
        while True:
            try:
                img = self.create_text_image(index), index
                return img
            except Exception:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                traceback.print_exc()
                print("error font", "unicode list", self.cur_unicode_list,
                      "cur text image params", self.cur_params,
                      "cur text", self.cur_text_ing_before,
                      "changed text", self.cur_text_ing,
                      "font", self.cur_font,
                      "font_size", self.cur_font_size)
                self.__before_hook__()

                tmp_index = random.choice(list(range(len(self.font_list))))
                while tmp_index == index:
                    tmp_index = random.choice(list(range(len(self.font_list))))
                index = tmp_index
                # print("error font", self.font_list[class_idx])
                1

    def __len__(self):
        return len(self.font_list) * self.num_sample_each_class


class CoupleOnlineRandomDataset(Dataset):
    CHOSUNG_SIM_LIST = {
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄲ': ['ㄱ'],
        'ㄴ': ['ㄴ', 'ㄷ'],
        'ㄷ': ['ㄸ', 'ㅁ', 'ㅌ'],
        'ㄸ': ['ㄷ', 'ㅁ', 'ㅌ'],
        'ㄹ': ['ㄷ'],
        'ㅁ': ['ㅇ', 'ㅍ', 'ㅂ'],
        'ㅂ': ['ㅁ'],
        'ㅃ': ['ㅂ', 'ㅁ'],
        'ㅅ': ['ㅆ', 'ㅈ'],
        'ㅆ': ['ㅅ', 'ㅉ'],
        'ㅇ': ['ㅁ', 'ㅎ'],
        'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ'],
        'ㅉ': ['ㅆ', 'ㅈ'],
        'ㅊ': ['ㅉ', 'ㅈ'],
        'ㅋ': ['ㄱ', 'ㄲ'],
        'ㅌ': ['ㄷ'],
        'ㅍ': ['ㅁ', 'ㅂ'],
        'ㅎ': ['ㅇ']
    }
    JUNGSUNG_SIM_LIST = {
        'ㅏ': ['ㅑ', 'ㅣ'],
        'ㅐ': ['ㅣ', 'ㅒ', 'ㅔ'],
        'ㅑ': ['ㅏ', 'ㅣ'],
        'ㅒ': ['ㅐ', 'ㅖ'],
        'ㅓ': ['ㅕ', 'ㅣ', 'ㅔ'],
        'ㅔ': ['ㅓ', 'ㅖ', 'ㅐ'],
        'ㅕ': ['ㅓ', 'ㅣ'],
        'ㅖ': ['ㅕ', 'ㅐ'],
        'ㅗ': ['ㅡ', 'ㅛ'],
        'ㅘ': ['ㅚ'],
        'ㅙ': ['ㅚ'],
        'ㅚ': ['ㅘ', 'ㅗㅒ'],
        'ㅛ': ['ㅗ'],
        'ㅜ': ['ㅠ', 'ㅡ'],
        'ㅝ': ['ㅞ', 'ㅟ'],
        'ㅞ': ['ㅝ'],
        'ㅟ': ['ㅝ'],
        'ㅠ': ['ㅜ'],
        'ㅡ': ['ㅜ', 'ㅗ'],
        'ㅢ': ['ㅝ'],
        'ㅣ': ['ㅏ', 'ㅓ'],
    }

    JONGSUNG_SIM_LIST = {
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄲ': ['ㄱ'],
        'ㄳ': ['ㄲ'],
        'ㄴ': ['ㄴ', 'ㄷ'],
        'ㄵ': ['ㄶ'],
        'ㄶ': ['ㄵ'],
        'ㄷ': ['ㄸ', 'ㅁ', 'ㅌ'],
        'ㄹ': ['ㄷ'],
        'ㄺ': ['ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄻ': ['ㄺ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄼ': ['ㄺ', 'ㄻ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄽ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄾ', 'ㄿ', 'ㅀ'],
        'ㄿ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㅀ'],
        'ㄾ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄿ', 'ㅀ'],
        'ㅀ': ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ'],
        'ㅁ': ['ㅇ', 'ㅍ', 'ㅂ'],
        'ㅂ': ['ㅁ'],
        'ㅄ': ['ㄽ'],
        'ㅅ': ['ㅆ', 'ㅈ'],
        'ㅆ': ['ㅅ', 'ㅉ'],
        'ㅇ': ['ㅁ', 'ㅎ'],
        'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ'],
        'ㅊ': ['ㅉ', 'ㅈ'],
        'ㅋ': ['ㄱ', 'ㄲ'],
        'ㅌ': ['ㄷ'],
        'ㅍ': ['ㅁ', 'ㅂ'],
        'ㅎ': ['ㅇ']
    }

    def __init__(self, font_list, transform=None, simple_transform=None, generation_params=None, bg_list=None,
                 num_sample_each_class=1000,
                 min_chars=1, max_chars=3,
                 hangul_unicodes=[44036, 44039, 44040], eng_unicodes=None,
                 number_unicodes=None,
                 hangul_prob=0.4, eng_prob=0.3, num_prob=0.3, mix_prob=0.5, change_prob=0.25,
                 min_change_char_ratio=0.33, max_change_char_ratio=1.0, hangul_change_uni_range=10,
                 use_augmentation=True,
                 use_complex_data=True):
        self.font_list = font_list

        classes = [os.path.splitext(os.path.basename(font_path))[0] for font_path in font_list]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_indices = list(range(len(classes)))
        self.classes = classes
        self.class_indices = class_indices
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.simple_transform = simple_transform
        self.bg_list = bg_list
        self.num_sample_each_class = num_sample_each_class
        self.generator_param = TextImageParamParser(generation_params)
        self.cur_unicode_list = None
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
        self.change_prob = change_prob
        self.min_change_char_ratio = min_change_char_ratio
        self.max_change_char_ratio = max_change_char_ratio
        self.hangul_change_uni_range = hangul_change_uni_range
        self.cur_text_ing = None
        self.cur_text_ing_before = None
        self.cur_is_simple = None
        self.cur_char_params = None

        self.use_augmentation = use_augmentation
        self.use_complex_data = use_complex_data

    def _sampling_text(self):
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
        self.cur_unicode_list = result_unicode_list
        self.cur_text_ing_before = "".join([chr(c) for c in self.cur_unicode_list])

    def __before_hook__(self):
        self._sampling_text()

    def get_classes(self):
        return self.classes

    def _get_random_bg_img(self):
        return random.choice(self.bg_list)

    def create_text_image(self, font_index, is_simple=True, target_width=None, target_height=None):
        import hgtk

        cur_unicode_list = []
        if random.random() <= self.change_prob:
            change_char_ratio = random.uniform(self.min_change_char_ratio, self.max_change_char_ratio)
            num_change = int(len(self.cur_unicode_list) * change_char_ratio)
            change_idx_list = list(range(len(self.cur_unicode_list)))
            random.shuffle(change_idx_list)
            if num_change < 1:
                num_change = 1
            change_idx_list = change_idx_list[:num_change]

            for i, uni_num in enumerate(self.cur_unicode_list):
                if i in change_idx_list:
                    if self.hangul_unicodes[0] <= uni_num <= self.hangul_unicodes[-1]:
                        if random.random() <= 0.5:
                            dec = hgtk.letter.decompose(chr(uni_num))
                            chosung = dec[0]
                            jungsung = dec[1]
                            jongsung = dec[2]
                            chosung = random.choice(MatchOnlineRandomDataset.CHOSUNG_SIM_LIST[chosung])
                            jungsung = random.choice(MatchOnlineRandomDataset.JUNGSUNG_SIM_LIST[jungsung])

                            if jongsung == '':
                                if random.random() > 0.8:
                                    k = random.choice(list(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST.keys()))
                                    jongsung = random.choice(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST[k])
                            else:
                                if random.random() > 0.8:
                                    jongsung = ''
                                else:
                                    jongsung = random.choice(MatchOnlineRandomDataset.JONGSUNG_SIM_LIST[jongsung])
                            try:
                                tmp_uni_num = ord(hgtk.letter.compose(chosung, jungsung, jongsung))
                            except:
                                tmp_uni_num = None
                            if tmp_uni_num is None or tmp_uni_num not in self.hangul_unicodes:
                                cur_unicode_list.append(uni_num)
                            else:
                                cur_unicode_list.append(tmp_uni_num)
                        else:
                            uni_idx = self.hangul_unicodes.index(uni_num)
                            min_idx = uni_idx - self.hangul_change_uni_range
                            max_idx = uni_idx + self.hangul_change_uni_range
                            if min_idx < 0:
                                min_idx = 0
                            if max_idx > len(self.hangul_unicodes):
                                max_idx = len(self.hangul_unicodes)

                            cur_unicode_list.append(self.hangul_unicodes[random.randrange(min_idx, max_idx)])

                    elif self.number_unicodes[0] <= uni_num <= self.number_unicodes[-1]:
                        cur_unicode_list.append(random.choice(self.number_unicodes))
                    elif self.eng_unicodes[0] <= uni_num <= self.eng_unicodes[-1]:
                        cur_unicode_list.append(random.choice(self.eng_unicodes))
                    else:
                        raise Exception("error unicode range")
                else:
                    cur_unicode_list.append(uni_num)
        else:
            cur_unicode_list = self.cur_unicode_list
        self.cur_is_simple = is_simple

        text = "".join([chr(c) for c in cur_unicode_list])
        self.cur_text_ing = text
        use_binarize = False
        color_mode = "RGB"
        char_params = {"use_binarize": use_binarize, "color_mode": color_mode}
        if is_simple:
            font_size = self.cur_char_params["font_size"]
            self.cur_char_params = None
            # char_params['bg_color'] = (255, 255, 255, 255)
            # char_params['use_bg_color'] = True
            # char_params['paddings'] = {"left": 2, "top": 2, "right": 2, "bottom": 2}
            # char_params['fg_color'] = (0, 0, 0, 255)
            # char_params['text_gradient'] = None
            img = text_image_maker.create_simple_text_image_by_size(text, self.font_list[font_index],
                                                                    width=target_width,
                                                                    height=target_height, allowable_pixels=1,
                                                                    start_font_size=font_size)
        else:
            bg_params = self.generator_param.get_bg_param()
            if "pos_ratio" in bg_params:
                char_params["bg_img_path"] = self._get_random_bg_img()
            padding_params = self.generator_param.get_paddings_param()
            text_params = self.generator_param.get_text_param()
            char_params.update(bg_params)
            char_params.update(padding_params)
            char_params.update(text_params)

            char_params['font_size'] = random.randint(10, 200)

            self.cur_font = self.font_list[font_index]
            # import uuid
            # char_params['output_path'] = "../testimage/{}_{}_{}.jpg".format(self.cur_unicode_list[0], index,
            #                                                                 str(uuid.uuid4()))
            # text_image_maker.create_text_image(text, self.font_list[index], **char_params)
            char_params['output_path'] = None
            self.cur_char_params = char_params
            if self.use_complex_data:
                img = text_image_maker.create_text_image(text, self.font_list[font_index], **char_params)
            else:
                img = text_image_maker.create_simple_text_image_by_size(text, self.font_list[font_index],
                                                                        width=None,
                                                                        height=None, allowable_pixels=1,
                                                                        start_font_size=char_params['font_size'])
        # todo: crop max width size
        height = img.shape[0]
        width = img.shape[1]
        if is_simple:
            if self.simple_transform is not None:
                img = self.simple_transform(image=img)['image']
        else:
            if self.transform is not None and self.use_augmentation:
                img = self.transform(image=img)['image']
            else:
                img = self.simple_transform(image=img)['image']

        return img, width, height

    def _random_font_index(self):
        return random.randint(0, len(self.font_list) - 1)

    def get(self, font_index, label):
        while True:
            try:
                if label:
                    real_font_index = font_index
                    fake_font_index = font_index
                else:
                    fake_font_index = self._random_font_index()
                    real_font_index = font_index
                    while fake_font_index == real_font_index:
                        fake_font_index = self._random_font_index()

                self._sampling_text()
                real_img, width, height = self.create_text_image(real_font_index, False)
                fake_img, _, _ = self.create_text_image(fake_font_index, True, width, height)
                return real_img, fake_img
            except OSError as e:
                if isinstance(e.args[0], str) and e.args[0].startswith("cannot identify image file"):
                    if os.path.isfile(self.cur_char_params["bg_img_path"]):
                        os.unlink(self.cur_char_params["bg_img_path"])
            except Exception:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                traceback.print_exc()
                # print("error font", self.font_list[class_idx])
                print("error font", "label", label, "unicode list", self.cur_unicode_list, "changed text",
                      self.cur_text_ing, "before text", self.cur_text_ing_before, "real font",
                      self.font_list[real_font_index],
                      "fake font", self.font_list[fake_font_index],
                      "cur is simple", self.cur_is_simple,
                      "cur text image params", self.cur_char_params)
                pass

    def __getitem__(self, index):
        label = int(random.random() < 0.5)
        while True:
            try:
                fake_font_index = self._random_font_index()
                if label:
                    real_font_index = fake_font_index
                else:
                    real_font_index = self._random_font_index()
                    while fake_font_index == real_font_index:
                        real_font_index = self._random_font_index()
                self._sampling_text()
                real_img, width, height = self.create_text_image(real_font_index, False)
                fake_img, _, _ = self.create_text_image(fake_font_index, True, width, height)
                return real_img, fake_img, float(label)
            except OSError as e:
                if isinstance(e.args[0], str) and e.args[0].startswith("cannot identify image file"):
                    if os.path.isfile(self.cur_char_params["bg_img_path"]):
                        os.unlink(self.cur_char_params["bg_img_path"])
                continue
            except Exception:
                # print(self.cur_text, self.cur_font, self.cur_params, traceback.format_exc())
                # traceback.print_exc()
                # print("exception font indexes", real_font_index, fake_font_index)
                # print("error font", "label", label, "unicode list", self.cur_unicode_list, "changed text",
                #       self.cur_text_ing, "before text", self.cur_text_ing_before, "real font",
                #       self.font_list[real_font_index],
                #       "fake font", self.font_list[fake_font_index],
                #       "cur is simple", self.cur_is_simple,
                #       "cur text image params", self.cur_char_params)
                pass

    def __len__(self):
        return len(self.font_list) * self.num_sample_each_class // 2


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


class OnlineTextColorDataset(HookDataset):
    def __init__(self, font_list, transform=None, generation_params=None, bg_list=None, num_sample_each_class=100,
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
                 change_font_in_error=True):
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
            a = 0.05 * ((random.random()) - (1 / 2))
            b = 0.1 * ((random.random()) - (1 / 2))
            char_params["img_persp_trans_params"] = [a, b]
        if random.random() <= self.use_text_persp_trans_prob:
            char_params["use_text_persp_trans"] = True
            a = 0.05 * ((random.random()) - (1 / 2))
            b = 0.1 * ((random.random()) - (1 / 2))
            char_params["text_persp_trans_params"] = [a, b]

        return char_params

    def create_text_image(self, index):
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
        # import json
        # print(json.dumps(char_params))
        # char_params = json.loads('{"color_mode": "RGB", "paddings": {"right": 0.1050272078196586, "top": 0.26090272932922254, "bottom": 0.17015320517900254, "left": 0.25837411687366774}, "text_border": null, "fg_color": [209, 235, 98, 166], "text_italic": false, "font_size": 15, "bg_img_path": "/home/irelin/resource/font_recognition/bgs/294.winterwax-500x500.jpg", "text_shadow": null, "use_img_persp_trans": true, "pos_ratio": [0.3, 0.2], "text_persp_trans_params": [-0.009976076882772873, 0.036253351174196216], "text_rotate": 14, "bg_img_width_ratio": 1.2, "text_blur": 0, "bg_img_height_ratio": 1.1, "use_bg_color": false, "bg_img_scale": 0.5601430892743575, "use_text_persp_trans": true, "use_binarize": false, "img_persp_trans_params": [-0.02296148027430462, 0.004213654695530833], "text_width_ratio": 1.0, "text_gradient": null, "output_path": null, "auto_chance_color_when_same": true, "text_height_ratio": 1.0}')
        # if self.use_debug:
        #     print(os.path.basename(self.font_list[index]), text)
        img = text_image_maker.create_text_image(text, self.font_list[index], **char_params)

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

        # return img, char_params['fg_color'][0], char_params['fg_color'][1], char_params['fg_color'][2]  # , \
        # char_params['bg_color'][0], char_params['bg_color'][1], char_params['bg_color'][2]
        return img, int((char_params['fg_color'][0] + 1) / 8), int((char_params['fg_color'][1] + 1) / 8), int(
            (char_params['fg_color'][2] + 1) / 8)

    def _get_random_index(self):
        return random.choice(list(range(len(self.font_list))))

    def __getitem__(self, index):
        index = self._get_random_index()
        while True:
            try:
                # if self.use_debug:
                #     print("start font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
                return self.create_text_image(index)
                # if self.use_debug:
                #     print("end font synth", self.font_list[index], "".join([chr(c) for c in self.cur_unicode_list]))
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
        return len(self.font_list) * self.num_sample_each_class
