from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np
import cv2
import math
import copy
from albumentations import IAAAffine, IAAPerspective
import random

angle_map = {"left": 225, "vertical": 270, "right": 315, "top": 45, "horizontal": 0, "down": 315}


def get_text_size(font, char):
    return font.getsize(char)
    # im = Image.new("L", (1, 1), "black")
    # draw = ImageDraw.Draw(im)
    # return draw.textsize(char, font)


def get_image_each_empty_row_nums(im):
    bbox = im.getbbox()
    w, h = im.size
    if not bbox:
        return int(h * 0.1), int(h * 0.1), int(w * 0.1), int(w * 0.1)
    return bbox[1], h - bbox[3], bbox[0], w - bbox[2]

    # im_array = np.array(im)
    # top_row_cnt = 0
    # for row in im_array:
    #     if row.sum() == 0:
    #         top_row_cnt += 1
    #     else:
    #         break
    #
    # bot_row_cnt = 0
    # for i in range(len(im_array) - 1, -1, -1):
    #     if im_array[i].sum() == 0:
    #         bot_row_cnt += 1
    #     else:
    #         break
    #
    # left_row_cnt = 0
    # for i in range(len(im_array[0])):
    #     if im_array[:, i].sum() == 0:
    #         left_row_cnt += 1
    #     else:
    #         break
    #
    # right_row_cnt = 0
    # for i in range(len(im_array[0]) - 1, -1, -1):
    #     if im_array[:, i].sum() == 0:
    #         right_row_cnt += 1
    #     else:
    #         break
    #
    # return top_row_cnt, bot_row_cnt, left_row_cnt, right_row_cnt


def get_each_empty_row_nums(char, font, img_width, img_height, text_width,
                            text_height, x=None, y=None):
    if x is None:
        x = (img_width - text_width) / 2
    if y is None:
        y = (img_height - text_height) / 2

    im = render_text(font, char, x, y, img_width, img_height, text_color=(255, 255, 255, 255))

    result = get_image_each_empty_row_nums(im)
    im.close()
    im = None
    return result


def interpolate(startValue, endValue, stepNumber, lastStepNumber):
    return ((endValue - startValue) * stepNumber / lastStepNumber + startValue).astype(int)


def get_masks(font, text, char_spacing=0):
    tmp_mask = font.getmask(text, mode="L")
    tmp_w, tmp_h = tmp_mask.size

    margin_ratio = 1.5

    tmp_im = Image.new('L', (int(tmp_w * margin_ratio), tmp_h), 'black')
    tmp_draw = ImageDraw.Draw(tmp_im)
    tmp_draw.text((0, 0), text[0], fill='white', font=font)

    tmp_mask_im = Image.new('L', (int(tmp_w * margin_ratio), tmp_h), 'black')
    tmp_mask_draw = ImageDraw.Draw(tmp_mask_im)
    tmp_mask_draw.text((0, 0), text[0], fill='white', font=font)
    tmp_mask_im = np.array(tmp_mask_im)
    tmp_mask_im = tmp_mask_im > 0
    tmp_mask_im = tmp_mask_im.astype(np.uint8) * 255

    tmp_mask_im = Image.fromarray(tmp_mask_im, mode="L")

    mask_im = Image.new('L', (int(tmp_w * margin_ratio), tmp_h), 'black')
    mask_im.paste(tmp_mask_im, (0, 0))
    tmp_mask_im.close()

    x1, y1, x2, y2 = tmp_im.getbbox()

    for i, char in enumerate(text[1:]):
        tmp_mask_im = Image.new('L', (int(tmp_w * margin_ratio) - x2, tmp_h), 'black')
        tmp_mask_draw = ImageDraw.Draw(tmp_mask_im)
        tmp_mask_draw.text((0, 0), char, fill='white', font=font)
        tmp_mask_im = np.array(tmp_mask_im)
        tmp_mask_im = tmp_mask_im > 0
        tmp_mask_im = tmp_mask_im.astype(np.uint8) * (254 - i)

        tmp_mask_im = Image.fromarray(tmp_mask_im, mode="L")
        mask_im.paste(tmp_mask_im, (x2 + char_spacing, 0))

        tmp_draw.text((x2 + char_spacing, 0), char, fill='white', font=font)
        x1, y1, x2, y2 = tmp_im.getbbox()
    x1, y1, x2, y2 = tmp_im.getbbox()
    tmp_im = tmp_im.crop((x1, y1, x2, y2))
    mask_im = mask_im.crop((x1, y1, x2, y2))
    ori_text_mask = tmp_im
    return ori_text_mask, mask_im


def aug_multi_masks(mask_im, result_im, aug):
    min_val = np.unique(mask_im)[1]
    mask_ar = np.array(mask_im)
    mask_channels = None
    for i in range(min_val, 256):
        tmp_channel = (mask_ar == i) * 1
        tmp_channel = np.expand_dims(tmp_channel, axis=0).astype(np.uint8)
        if mask_channels is None:
            mask_channels = tmp_channel
        else:
            mask_channels = np.concatenate((mask_channels, tmp_channel), axis=0)
    result = aug(image=np.array(result_im), masks=mask_channels)
    image = result['image']
    mask_sum = None
    for i, tmp_mask in enumerate(result['masks']):
        tmp_channel = tmp_mask * (min_val + i)
        if mask_sum is None:
            mask_sum = tmp_channel
        else:
            union_mask = mask_sum & tmp_channel
            if np.sum(union_mask) > 0:
                mask_sum *= (union_mask == False)

            mask_sum += tmp_channel

    return image, mask_sum


def create_text_shadow(text_shadow, image_width, image_height, ori_text_mask, mask_width, mask_height, text_x, text_y):
    shadow_color = tuple(text_shadow["color"])
    shadow_width = text_shadow["width"]
    shadow_blur_amount = text_shadow["blur_count"]
    direction = text_shadow["direction"]

    mask_list = []
    for i in range(shadow_width):
        gap = i + 1
        if direction == "bottom_right":
            tmp_mask1 = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y + gap, image_width,
                                 image_height, is_expand=True)
            tmp_mask2 = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y, image_width,
                                 image_height, is_expand=True)
            tmp_mask3 = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y + gap, image_width,
                                 image_height, is_expand=True)
            if tmp_mask1 is not None:
                mask_list.append(tmp_mask1)
            if tmp_mask2 is not None:
                mask_list.append(tmp_mask2)
            if tmp_mask3 is not None:
                mask_list.append(tmp_mask3)
        elif direction == "right":
            tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y, image_width,
                                image_height, is_expand=True)
            if tmp_mask is not None:
                mask_list.append(tmp_mask)
        elif direction == "top_right":
            tmp_mask1 = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y - gap, image_width,
                                 image_height, is_expand=True)
            tmp_mask2 = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y, image_width,
                                 image_height, is_expand=True)
            tmp_mask3 = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y - gap, image_width,
                                 image_height, is_expand=True)
            if tmp_mask1 is not None:
                mask_list.append(tmp_mask1)
            if tmp_mask2 is not None:
                mask_list.append(tmp_mask2)
            if tmp_mask3 is not None:
                mask_list.append(tmp_mask3)
        elif direction == "top":
            tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y - gap, image_width,
                                image_height, is_expand=True)
            if tmp_mask is not None:
                mask_list.append(tmp_mask)
        elif direction == "top_left":
            tmp_mask1 = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y, image_width,
                                 image_height, is_expand=True)
            tmp_mask2 = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y - gap, image_width,
                                 image_height, is_expand=True)
            tmp_mask3 = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y - gap, image_width,
                                 image_height, is_expand=True)
            if tmp_mask1 is not None:
                mask_list.append(tmp_mask1)
            if tmp_mask2 is not None:
                mask_list.append(tmp_mask2)
            if tmp_mask3 is not None:
                mask_list.append(tmp_mask3)
        elif direction == "left":
            tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y, image_width,
                                image_height, is_expand=True)
            if tmp_mask is not None:
                mask_list.append(tmp_mask)
        elif direction == "bottom_left":
            tmp_mask1 = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y + gap, image_width,
                                 image_height, is_expand=True)
            tmp_mask2 = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y, image_width,
                                 image_height, is_expand=True)
            tmp_mask3 = pad_mask(ori_text_mask, mask_height, mask_width, text_x - + gap, text_y + gap, image_width,
                                 image_height, is_expand=True)
            if tmp_mask1 is not None:
                mask_list.append(tmp_mask1)
            if tmp_mask2 is not None:
                mask_list.append(tmp_mask2)
            if tmp_mask3 is not None:
                mask_list.append(tmp_mask3)

        elif direction == "bottom":
            tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y + gap, image_width,
                                image_height, is_expand=True)
            if tmp_mask is not None:
                mask_list.append(tmp_mask)

    if len(mask_list) > 0:
        if len(mask_list) == 1:
            tmp_mask = mask_list[0]
        else:
            tmp_mask = np.concatenate(mask_list, axis=2)
            tmp_mask = np.amax(tmp_mask, axis=2)
            tmp_mask = np.expand_dims(tmp_mask, axis=2)
        shadow_im = render_text_masked(tmp_mask, image_width, image_height, text_color=shadow_color)
        mask_list = None
        tmp_mask = None

        if shadow_blur_amount > 0:
            for i in range(shadow_blur_amount):
                shadow_im = shadow_im.filter(ImageFilter.BLUR)

    return shadow_im


def create_text_border(text_border, image_width, image_height, ori_text_mask, mask_width, mask_height, text_x, text_y):
    border_color = tuple(text_border["color"])
    border_width = text_border["width"]
    border_blur_iter_num = text_border["blur_count"]

    mask_list = []
    for i in range(border_width):
        gap = i + 1

        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y - gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x, text_y + gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y - gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y - gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x - gap, text_y + gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)
        tmp_mask = pad_mask(ori_text_mask, mask_height, mask_width, text_x + gap, text_y + gap, image_width,
                            image_height, is_expand=True)
        if tmp_mask is not None:
            mask_list.append(tmp_mask)

    if len(mask_list) > 0:
        if len(mask_list) == 1:
            tmp_mask = mask_list[0]
        else:
            tmp_mask = np.concatenate(mask_list, axis=2)
            tmp_mask = np.amax(tmp_mask, axis=2)
            tmp_mask = np.expand_dims(tmp_mask, axis=2)
        border_im = render_text_masked(tmp_mask, image_width, image_height, text_color=border_color)
        mask_list = None
        tmp_mask = None

        if border_blur_iter_num > 0:
            for i in range(border_blur_iter_num):
                border_im = border_im.filter(ImageFilter.BLUR)

    return border_im


def draw_text(text_x, text_y, text, fg_color, font, text_border=None, text_gradient=None, text_skew=None,
              text_shadow=None, text_width_ratio=1.0, text_height_ratio=1.0, text_rotate=0.0,
              image_width=None, image_height=None, char_spacing=0, text_italic=None, italic_ratio=0.0,
              use_text_persp_trans=False, text_persp_trans_params=None, raise_exception=False,
              use_default_render=False, return_mask=False):
    if return_mask:
        ori_text_mask, mask_im = get_masks(font, text, char_spacing)
    else:
        ori_text_mask = font.getmask(text)
    mask_width, mask_height = ori_text_mask.size

    shadow_im = None
    border_im = None

    if text_shadow is not None:
        shadow_im = create_text_shadow(text_shadow, image_width, image_height, ori_text_mask, mask_width, mask_height,
                                       text_x, text_y)

    if text_border is not None:
        border_im = create_text_border(text_border, image_width, image_height, ori_text_mask, mask_width, mask_height,
                                       text_x, text_y)

    if text_gradient is None or len(text_gradient["anchors"]) < 2:
        if use_default_render:
            text_im = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_im)
            draw.text((text_x, text_y), text, fill=fg_color, font=font)
        else:
            text_im = render_text(font, text, text_x, text_y, image_width, image_height, text_color=fg_color,
                                  text_mask=ori_text_mask)

            if return_mask:
                tmp_mask_im = Image.new('L', (image_width, image_height), 'black')
                tmp_mask_im.paste(mask_im, (text_x, text_y))
                mask_im = tmp_mask_im

    elif text_gradient["type"] == "linear":
        grad_array = get_multiple_gradation(image_width, image_height, text_gradient)
        text_im = render_text(font, text, text_x, text_y, image_width, image_height, draw_im=grad_array,
                              text_mask=ori_text_mask)

        if return_mask:
            tmp_mask_im = Image.new('L', (image_width, image_height), 'black')
            tmp_mask_im.paste(mask_im, (text_x, text_y))
            mask_im = tmp_mask_im
    else:
        if raise_exception:
            raise Exception("no text foreground.")

    if shadow_im is not None:
        result_im = Image.alpha_composite(shadow_im, text_im)
        text_im.close()
        text_im = None
        shadow_im.close()
        shadow_im = None
    else:
        result_im = text_im

    if border_im is not None:
        result_im = Image.alpha_composite(border_im, result_im)
        border_im.close()
        border_im = None

    if text_italic:
        width, height = result_im.size
        m = italic_ratio
        xshift = abs(m) * width
        new_width = width + int(round(xshift))
        result_im = result_im.transform((new_width, height), Image.AFFINE,
                                        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

        if return_mask:
            mask_im = mask_im.transform((new_width, height), Image.AFFINE,
                                        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.NEAREST)

    if use_text_persp_trans:
        result_im = perspective_transform(result_im, limit_ratio=0.2, params=text_persp_trans_params,
                                          inter=Image.BICUBIC)
        if return_mask:
            mask_im = perspective_transform(mask_im, limit_ratio=0.2, params=text_persp_trans_params,
                                            inter=Image.NEAREST)

    if text_width_ratio > 0.:
        w, h = result_im.size
        result_im = result_im.resize((int(w * text_width_ratio), h), resample=Image.BILINEAR)
        if return_mask:
            mask_im = mask_im.resize((int(w * text_width_ratio), h), resample=Image.NEAREST)

    if text_height_ratio > 0.:
        w, h = result_im.size
        result_im = result_im.resize((w, int(h * text_height_ratio)), resample=Image.BILINEAR)
        if return_mask:
            mask_im = mask_im.resize((w, int(h * text_height_ratio)), resample=Image.NEAREST)

    if text_rotate is not None and (text_rotate > -360 and text_rotate < 360):
        if result_im.size[0] > result_im.size[1]:
            tmp_im = Image.new("RGBA", (result_im.size[0], result_im.size[0]), (0, 0, 0, 0))
            tmp_im.paste(result_im, (0, round(result_im.size[0] / 2) - round(result_im.size[1] / 2)))
            result_im = tmp_im
            if return_mask:
                tmp_im = Image.new("L", (mask_im.size[0], mask_im.size[0]), 'black')
                tmp_im.paste(mask_im, (0, round(mask_im.size[0] / 2) - round(mask_im.size[1] / 2)))
                mask_im = tmp_im
        result_im = result_im.rotate(text_rotate, resample=Image.BILINEAR)
        if return_mask:
            mask_im = mask_im.rotate(text_rotate, resample=Image.NEAREST)

    if return_mask:
        return result_im, mask_im
    else:
        return result_im


def get_bbox_area(im):
    bbox = im.getbbox()
    if bbox is None:
        return 0

    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _perspective_transform(image, return_arr=False, params=None, mask=None):
    # if params:
    #     a = params[0]
    #     b = params[1]
    # else:
    #     a = 0.05 * (random.random() - (1 / 2))
    #     b = 0.1 * (random.random() - (1 / 2))
    #
    if params:
        aug = params
    else:
        a = 0.05 * (random.random() - (1 / 2))
        b = 0.1 * (random.random() - (1 / 2))
        aug = IAAPerspective(p=1, scale=(0.05 + a, 0.1 + b), keep_size=True)

    if mask:
        image, mask_im = aug_multi_masks(mask, image, aug)

        if return_arr:
            return Image.fromarray(image), image, Image.fromarray(mask_im.astype(np.uint8)), mask_im
        else:
            return Image.fromarray(image), Image.fromarray(mask_im.astype(np.uint8))
    else:
        image = aug(image=np.array(image))['image']
        if return_arr:
            return Image.fromarray(image), image
        else:
            return Image.fromarray(image)


def _perspective_transform_with_bg(image_with_bg, img_persp_trans_params, mask=None):
    ori_std = np.array(image_with_bg).std()
    if mask:
        image, image_ar, tmp_mask, mask_ar = _perspective_transform(image_with_bg, True, params=img_persp_trans_params,
                                                                    mask=mask)
    else:
        image, image_ar = _perspective_transform(image_with_bg, True, params=img_persp_trans_params)
    retry_cnt = 0
    while image.getbbox() is None or image_ar.std() < ori_std / 4:
        retry_cnt += 1
        if retry_cnt > 2:
            if mask is not None:
                return image_with_bg, mask
            else:
                return image_with_bg
            # raise Exception("image perspective error")
        if mask:
            image, image_ar, tmp_mask, mask_ar = _perspective_transform(image_with_bg, True,
                                                                        params=img_persp_trans_params,
                                                                        mask=mask)
        else:
            image, image_ar = _perspective_transform(image_with_bg, True, params=img_persp_trans_params)

    if mask is not None:
        mask = tmp_mask
        return image, mask
    return image


def pad_mask(text_mask, mask_height, mask_width, text_x=0, text_y=0, image_width=None, image_height=None,
             is_expand=True):
    if image_width is None:
        image_width = mask_width
    if image_height is None:
        image_height = mask_height

    if image_width < mask_width + text_x or image_height < mask_height + text_y:
        return None
    if text_x < 0:
        text_x = 0
    if text_y < 0:
        text_y = 0

    if not isinstance(text_mask, np.ndarray):
        text_mask = np.array(text_mask)
        text_mask = text_mask.reshape((mask_height, mask_width))
    text_mask = np.pad(text_mask,
                       ((text_y, image_height - mask_height - text_y), (text_x, image_width - mask_width - text_x)),
                       mode='constant')
    if is_expand:
        text_mask = np.expand_dims(text_mask, axis=2)
    return text_mask


def render_text(font, text, text_x=0, text_y=0, image_width=None, image_height=None, text_color=None, draw_im=None,
                text_mask=None):
    if not text_mask:
        text_mask = font.getmask(text)
    mask_width, mask_height = text_mask.size
    text_mask = pad_mask(text_mask, mask_height, mask_width, text_x, text_y, image_width, image_height)
    if text_mask is None:
        raise Exception("failed to pad text mask")
    if draw_im is None:
        if text_color is None:
            text_color = (0, 0, 0, 0)
        draw_im = Image.new("RGBA", (image_width, image_height), tuple(text_color))  # (0, 0, 0, 0))
    if not isinstance(draw_im, np.ndarray):
        im_arr = np.array(draw_im)
    else:
        im_arr = draw_im
    im_alpha = im_arr[:, :, 3:4] * (text_mask / 255)
    im_rgb = im_arr[:, :, :3] * (text_mask != 0).astype(int)
    im_arr = np.concatenate((im_rgb, im_alpha), axis=2)
    draw_im = Image.fromarray(im_arr.astype(int).astype(np.int8), mode="RGBA")
    return draw_im


def render_text_masked(text_mask, image_width=None, image_height=None, text_color=None, draw_im=None):
    if draw_im is None:
        if text_color is None:
            text_color = (0, 0, 0, 0)
        draw_im = Image.new("RGBA", (image_width, image_height), tuple(text_color))  # (0, 0, 0, 0))
    im_arr = np.array(draw_im)
    im_alpha = im_arr[:, :, 3:4] * (text_mask / 255)
    im_rgb = im_arr[:, :, :3] * (text_mask != 0).astype(int)
    im_arr = np.concatenate((im_rgb, im_alpha), axis=2)
    draw_im = Image.fromarray(im_arr.astype(int).astype(np.int8), mode="RGBA")
    return draw_im


def get_font_size_by_text_height(font_path, text, text_height, allowable_pixels=0, font_size_step=5,
                                 max_try=30, max_allowable_pixels=3, start_font_size=None, min_font_size=4):
    paddings = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    font_size = round(text_height * 1.1)
    pad_length = text_height * paddings["top"] + text_height * paddings["bottom"]
    target_length = text_height

    if start_font_size:
        font_size = start_font_size

    is_small_target = None
    # font_size_step = int(font_size * 0.16)
    tried = 0
    min_diff_pixels = 1000000
    best_font_size = font_size
    last_try = max_try * 2
    while True:
        if font_size < min_font_size:
            font_size = 7
            break
            # raise Exception("min font size")
        font = ImageFont.truetype(font_path, font_size)

        text_mask = font.getmask(text)
        text_width, text_height = text_mask.size

        text_length = text_height

        cur_diff_pixels = abs(target_length - (text_length + pad_length))
        if cur_diff_pixels <= allowable_pixels:
            break

        if min_diff_pixels > cur_diff_pixels:
            min_diff_pixels = cur_diff_pixels
            best_font_size = font_size

        if target_length > (text_length + pad_length):
            if is_small_target:
                if font_size_step == 1:
                    break
                else:
                    font_size_step -= 1
                is_small_target = False
            if is_small_target is None:
                is_small_target = False

            font_size += font_size_step
        else:
            if is_small_target is not None and not is_small_target:
                if font_size_step == 1:
                    break
                else:
                    font_size_step -= 1
                is_small_target = True

            if is_small_target is None:
                is_small_target = True
            font_size -= font_size_step
        if tried > last_try:
            font_size = best_font_size
            break
        if tried > max_try:
            if min_diff_pixels <= max_allowable_pixels:
                font_size = best_font_size
                break
            else:
                max_try += 10
        tried += 1
    return font_size


def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def create_simple_text_image_by_size(text, font_path, width, height,
                                     paddings={"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05},
                                     allowable_pixels=0, font_size_step=5,
                                     output_path=None, fg_color=(0, 0, 0, 255), use_bg_color=True,
                                     bg_color=(255, 255, 255, 255),
                                     bg_img_path=None,
                                     image_quality=100, image_format="JPEG", limit_height=20,
                                     return_info=False,
                                     use_extend_bg_image=True,
                                     max_try=30,
                                     max_allowable_pixels=3, start_font_size=None, min_font_size=4,
                                     target_resize=True,
                                     raise_exception=True,
                                     return_numpy=True, use_default_render=False):
    if width is not None and height is not None:

        if width * 2 <= height:
            use_width = True
            font_size = int(width * 0.9)
            pad_length = paddings["left"] + paddings["right"]
            target_length = width
        else:
            use_width = False
            font_size = int(height * 0.9)
            pad_length = paddings["top"] + paddings["bottom"]
            target_length = height

        if start_font_size:
            font_size = start_font_size

        is_small_target = None
        # font_size_step = int(font_size * 0.16)
        tried = 0
        min_diff_pixels = 1000000
        best_font_size = font_size
        last_try = max_try * 2
        while True:
            if font_size < min_font_size:
                font_size = 7
                break
                # raise Exception("min font size")
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = get_text_size(font, text)

            if text_width > text_height:
                img_width = text_width * 2
                img_height = round(text_width * 1.5)
            else:
                img_width = round(text_height * 1.5)
                img_height = text_height * 2
            text_x = (img_width - text_width) // 2
            text_y = (img_height - text_height) // 2
            text_im = draw_text(text_x, text_y, text, fg_color, font, image_width=img_width, image_height=img_height,
                                raise_exception=raise_exception, use_default_render=use_default_render)

            img_width, img_height = text_im.size
            top, bottom, left, right = get_image_each_empty_row_nums(text_im)

            text_width = img_width - (right + left)
            text_height = img_height - (bottom + top)

            if use_width:
                text_length = text_width
            else:
                text_length = text_height

            cur_diff_pixels = abs(target_length - (text_length + pad_length))
            if cur_diff_pixels <= allowable_pixels:
                break

            if min_diff_pixels > cur_diff_pixels:
                min_diff_pixels = cur_diff_pixels
                best_font_size = font_size

            if target_length > (text_length + pad_length):
                if is_small_target:
                    if font_size_step == 1:
                        break
                    else:
                        font_size_step -= 1
                    is_small_target = False
                if is_small_target is None:
                    is_small_target = False

                font_size += font_size_step
            else:
                if is_small_target is not None and not is_small_target:
                    if font_size_step == 1:
                        break
                    else:
                        font_size_step -= 1
                    is_small_target = True

                if is_small_target is None:
                    is_small_target = True
                font_size -= font_size_step
            if tried > last_try:
                font_size = best_font_size
                break
            if tried > max_try:
                if min_diff_pixels <= max_allowable_pixels:
                    font_size = best_font_size
                    break
                else:
                    max_try += 10
            tried += 1
    else:
        font_size = start_font_size

    if target_resize:
        target_width = width
        target_height = height
    else:
        target_width = None
        target_height = None
    return create_text_image(text=text, font_path=font_path, font_size=font_size,
                             paddings=paddings,
                             output_path=output_path, fg_color=fg_color, use_bg_color=use_bg_color,
                             bg_color=bg_color,
                             bg_img_path=bg_img_path,
                             image_quality=image_quality, image_format=image_format, min_len=limit_height,
                             return_info=return_info,
                             use_extend_bg_image=use_extend_bg_image,
                             target_width=target_width,
                             target_height=target_height,
                             return_numpy=return_numpy)


def get_gradation_3d(width, height, start_list, stop_list, is_horizontal):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop) in enumerate(zip(start_list, stop_list)):
        result[:, :, i] = get_gradation_2d(start, stop, width, height, is_horizontal)

    return result


def get_multiple_gradation(target_width, target_height, grad_param):
    grad_array = None
    grad_param['anchors'] = sorted(grad_param['anchors'], key=lambda x: x['pos'])
    if grad_param['anchors'][0]['pos'] != 0:
        if grad_param['direction'] == 'horizontal':
            tw = round(target_width * grad_param['anchors'][0]['pos'])
            th = target_height
        else:
            tw = target_width
            th = round(target_height * grad_param['anchors'][0]['pos'])
        grad_array = np.full((th, tw, 4), grad_param['anchors'][0]['color'], dtype=np.uint8)

    for i in range(len(grad_param['anchors']) - 1):
        anchor = grad_param['anchors'][i]
        next_anchor = grad_param['anchors'][i + 1]
        if grad_param['direction'] == 'horizontal':
            tw = round(target_width * (next_anchor['pos'] - anchor['pos']))
            th = target_height
        else:
            th = round(target_height * (next_anchor['pos'] - anchor['pos']))
            tw = target_width
        tmp_grad = get_gradation_3d(tw, th, anchor['color'], next_anchor['color'],
                                    grad_param['direction'] == 'horizontal')
        if grad_array is None:
            grad_array = tmp_grad
        else:
            axis = 1 if grad_param['direction'] == 'horizontal' else 0
            grad_array = np.concatenate((grad_array, tmp_grad), axis=axis)

    if len(grad_array[0]) > target_width:
        grad_array = grad_array[:, :target_width]
    if len(grad_array) > target_height:
        grad_array = grad_array[:target_height]

    if len(grad_array[0]) < target_width or len(grad_array) < target_height:
        if grad_param['direction'] == 'horizontal':
            tw = target_width - len(grad_array[0])
            grad_array = np.concatenate(
                (grad_array, np.full((target_height, tw, 4), grad_param['anchors'][-1]['color'], dtype=np.uint8)),
                axis=1)
        else:
            th = target_height - len(grad_array)
            grad_array = np.concatenate(
                (grad_array, np.full((th, target_width, 4), grad_param['anchors'][-1]['color'], dtype=np.uint8)),
                axis=0)

    return grad_array


def perspective_transform(im, limit_ratio=0.45, params=None, inter=Image.BICUBIC):
    w, h = im.size
    x1, y1 = params[0]
    p_a1 = []
    p_b1 = []
    if x1 == 0:
        p_a1.append(0)
        p_b1.append(0)
    elif x1 > 0:
        p_a1.append(round(w * limit_ratio * abs(x1)))
        p_b1.append(0)
    else:
        p_a1.append(0)
        p_b1.append(round(w * limit_ratio * abs(x1)))

    if y1 == 0:
        p_a1.append(0)
        p_b1.append(0)
    elif y1 > 0:
        p_a1.append(round(h * limit_ratio * abs(y1)))
        p_b1.append(0)
    else:
        p_a1.append(0)
        p_b1.append(round(h * limit_ratio * abs(y1)))

    x2, y2 = params[1]
    p_a2 = []
    p_b2 = []
    if x2 == 0:
        p_a2.append(w)
        p_b2.append(w)
    elif x2 > 0:
        p_a2.append(w)
        p_b2.append(w - round(w * limit_ratio * abs(x2)))
    else:
        p_a2.append(w - round(w * limit_ratio * abs(x2)))
        p_b2.append(w)

    if y2 == 0:
        p_a2.append(0)
        p_b2.append(0)
    elif y2 > 0:
        p_a2.append(round(h * limit_ratio * abs(y2)))
        p_b2.append(0)
    else:
        p_a2.append(0)
        p_b2.append(round(h * limit_ratio * abs(y2)))

    x3, y3 = params[2]
    p_a3 = []
    p_b3 = []
    if x3 == 0:
        p_a3.append(w)
        p_b3.append(w)
    elif x3 > 0:
        p_a3.append(w)
        p_b3.append(w - round(w * limit_ratio * abs(x3)))
    else:
        p_a3.append(w - round(w * limit_ratio * abs(x3)))
        p_b3.append(w)

    if y3 == 0:
        p_a3.append(h)
        p_b3.append(h)
    elif y3 > 0:
        p_a3.append(h)
        p_b3.append(h - round(h * limit_ratio * abs(y3)))
    else:
        p_a3.append(h - round(h * limit_ratio * abs(y3)))
        p_b3.append(h)

    x4, y4 = params[3]
    p_a4 = []
    p_b4 = []
    if x4 == 0:
        p_a4.append(0)
        p_b4.append(0)
    elif x4 > 0:
        p_a4.append(round(w * limit_ratio * abs(x4)))
        p_b4.append(0)
    else:
        p_a4.append(0)
        p_b4.append(round(w * limit_ratio * abs(x4)))

    if y4 == 0:
        p_a4.append(h)
        p_b4.append(h)
    elif y4 > 0:
        p_a4.append(h)
        p_b4.append(h - round(h * limit_ratio * abs(y4)))
    else:
        p_a4.append(h - round(h * limit_ratio * abs(y4)))
        p_b4.append(h)

    coeffs = find_coeffs(
        [tuple(p_a1), tuple(p_a2), tuple(p_a3), tuple(p_a4)],
        [tuple(p_b1), tuple(p_b2), tuple(p_b3), tuple(p_b4)])

    return im.transform(im.size, Image.PERSPECTIVE, coeffs, inter)


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def create_composed_text_image(np_text_images, target_width, target_height, output_path=None,
                               use_bg_color=True,
                               bg_color=(255, 255, 255, 255),
                               bg_img_path=None, pos_ratio=None, bg_gradient=None, color_mode="RGB",
                               use_binarize=False, image_quality=100, image_format="JPEG",
                               use_extend_bg_image=True,
                               bg_img_scale=1.0,
                               bg_img_height_ratio=1.0, bg_img_width_ratio=1.0,
                               use_img_persp_trans=False,
                               img_persp_trans_params=None,
                               raise_exception=True, return_numpy=False,
                               target_x=None, target_y=None
                               ):
    if use_bg_color:
        if bg_gradient is None:
            bg_img = Image.new("RGBA", (target_width, target_height), tuple(bg_color))
        elif bg_gradient["type"] == "linear":
            grad_array = get_multiple_gradation(target_width, target_height, bg_gradient)
            bg_img = Image.fromarray(grad_array.astype(np.uint8), mode="RGBA")

    else:
        bg_img = resize_bg_image(bg_img_path, bg_img_scale, bg_img_width_ratio, bg_img_height_ratio, pos_ratio,
                                 use_extend_bg_image, target_width, target_height, raise_exception)

    texts_arr = np.zeros_like(bg_img, dtype=np.uint8)
    texts_mask = np.zeros((bg_img.size[1], bg_img.size[0]), dtype=np.uint8)

    text_image_pos_list = []
    none_cnt = 0
    for text_image in np_text_images:
        h_l_ratio = random.random()
        w_l_ratio = random.random()

        x1 = target_x if target_x is not None else round(w_l_ratio * target_width)
        y1 = target_y if target_y is not None else round(h_l_ratio * target_height)

        try_cnt = 0
        skip = False
        while np.sum(texts_mask[y1:y1 + text_image.shape[0], x1:x1 + text_image.shape[1]]) > 0 or x1 + text_image.shape[
            1] -1 >= target_width or y1 + text_image.shape[0] -1 >= target_height:
            try_cnt += 1
            if try_cnt == 10:
                skip = True
                break
            h_l_ratio = random.random()
            w_l_ratio = random.random()
            y1 = round(h_l_ratio * target_height)
            x1 = round(w_l_ratio * target_width)

        if skip:
            if target_x is not None and target_y is not None:
                raise Exception("target x and y is failed to make composed image.")
            text_image_pos_list.append(None)
            none_cnt += 1
            continue

        text_image_pos_list.append({"x": x1, "y": y1})

        texts_mask[y1:y1 + text_image.shape[0], x1:x1 + text_image.shape[1]] = np.ones(
            (text_image.shape[0], text_image.shape[1]), dtype=np.uint8)
        texts_arr[y1:y1 + text_image.shape[0], x1:x1 + text_image.shape[1], :] = text_image

    if len(np_text_images) == none_cnt:
        raise Exception("failed to insert all images")

    fg_img = Image.fromarray(texts_arr, mode="RGBA")

    im = Image.alpha_composite(bg_img, fg_img)

    if im.mode != color_mode:
        im = im.convert(color_mode)

    # if use_img_persp_trans:
    #     im = perspective_transform(im, limit_ratio=0.2, params=img_persp_trans_params,
    #                                inter=Image.BICUBIC)

    if use_binarize:
        img = np.array(im)
        img = img[:, :, ::-1].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im = Image.fromarray(img)
        img = None

    # if target_width and target_height:
    #     im = im.resize((target_width, target_height), resample=Image.BILINEAR)

    if output_path is None:
        if return_numpy:
            result = np.array(im)
            im.close()
            im = None
            return result, text_image_pos_list
        else:
            return im, text_image_pos_list
    else:
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        im.save(output_path, format=image_format, subsampling=0, quality=image_quality)
        im.close()
        im = None
        return text_image_pos_list


def resize_bg_image(bg_img_path, bg_img_scale, bg_img_width_ratio, bg_img_height_ratio, pos_ratio,
                    use_extend_bg_image, bg_w, bg_h, raise_exception):
    bg_img = Image.open(bg_img_path).convert("RGBA")

    bg_img_width, bg_img_height = bg_img.size

    if bg_img_height_ratio and bg_img_height_ratio != 1.0:
        bg_img_height *= bg_img_height_ratio
    if bg_img_width_ratio and bg_img_width_ratio != 1.0:
        bg_img_width *= bg_img_width_ratio

    if bg_img_scale and bg_img_scale != 1.0:
        bg_img_height *= bg_img_scale
        bg_img_width *= bg_img_scale

    bg_img_height = int(bg_img_height)
    bg_img_width = int(bg_img_width)
    bg_img = bg_img.resize((bg_img_width, bg_img_height))

    img_width = bg_img_width
    img_height = bg_img_height
    if pos_ratio:
        text_x = img_width * pos_ratio[0]
        text_y = img_height * pos_ratio[1]
    else:
        text_x = img_width
        text_y = img_height
    if use_extend_bg_image:
        if bg_img_width < bg_w + text_x or bg_img_height < bg_h + text_y:
            target_w = bg_w + text_x if bg_img_width < bg_w + text_x else bg_img_width
            target_h = bg_h + text_y if bg_img_height < bg_h + text_y else bg_img_height
            bg_img = bg_img.resize((int(target_w), int(target_h)))
    else:
        if bg_img_width < bg_w or bg_img_height < bg_h:
            if raise_exception:
                raise Exception("image bg size is more small than text. bg: {}x{}, text: {}x{}".format(
                    bg_img_width, bg_img_height, bg_w, bg_h
                ))

        img_width = bg_img_width
        img_height = bg_img_height
        if pos_ratio:
            text_x = img_width * pos_ratio[0]
            text_y = img_height * pos_ratio[1]
        else:
            text_x = img_width
            text_y = img_height
        if bg_img_width - text_x < bg_w:
            text_x = bg_img_width - bg_w
        if bg_img_height - text_y < bg_h:
            text_y = bg_img_height - bg_h

    return bg_img.crop((text_x, text_y, text_x + bg_w, text_y + bg_h))


def calc_paddings(paddings, text_width, text_height, min_len):
    paddings = copy.deepcopy(paddings)
    if text_height < min_len or text_width < min_len:
        if paddings["top"] < 0:
            paddings["top"] = 0
        if paddings["bottom"] < 0:
            paddings["bottom"] = 0
        if paddings["left"] < 0:
            paddings["left"] = 0
        if paddings["right"] < 0:
            paddings["right"] = 0

    paddings["left"] = round(text_height * paddings["left"])
    paddings["right"] = round(text_height * paddings["right"])
    paddings["top"] = round(text_height * paddings["top"])
    paddings["bottom"] = round(text_height * paddings["bottom"])
    if paddings["left"] + paddings["right"] <= -(text_width // 3):
        paddings["left"] = random.randint(-1, round(text_width * 0.4))
        paddings["right"] = random.randint(-1, round(text_width * 0.4))
    return paddings


def create_text_image(text, font_path, paddings={"left": 0.05, "top": 0.05, "right": 0.05, "bottom": 0.05},
                      font_size=160,
                      output_path=None, fg_color=(0, 0, 0, 255), use_bg_color=True, bg_color=(255, 255, 255, 255),
                      bg_img_path=None,
                      pos_ratio=None, text_rotate=None, text_blur=None, bg_gradient=None,
                      text_border=None, text_gradient=None, text_skew=None, text_italic=False, char_spacing=0,
                      text_shadow=None, text_width_ratio=1.0, text_height_ratio=1.0, color_mode="RGB",
                      use_binarize=False, image_quality=100, image_format="JPEG", min_len=20, return_info=False,
                      use_extend_bg_image=True, target_width=None, target_height=None, bg_img_scale=1.0,
                      bg_img_height_ratio=1.0, bg_img_width_ratio=1.0, italic_ratio=0.0,
                      use_text_persp_trans=False, use_img_persp_trans=False,
                      img_persp_trans_params=None, text_persp_trans_params=None,
                      auto_chance_color_when_same=False, raise_exception=True, return_numpy=True,
                      use_default_render=False,
                      return_mask=False):
    if use_bg_color and bg_gradient is None and (text_gradient is None or len(text_gradient["anchors"]) < 2) and tuple(
            bg_color) == tuple(fg_color):
        if auto_chance_color_when_same:
            bg_color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255), 255)
        else:
            if raise_exception:
                raise Exception("bg color and fg color are same", bg_color, fg_color)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = get_text_size(font, text)
    if text_width > text_height:
        if return_mask and char_spacing > 9:
            img_width = round(text_width * 2.)
            img_height = round(text_height * 2.)
        else:
            img_width = round(text_width * 1.5)
            img_height = round(text_height * 1.5)
    else:
        img_width = round(text_width * 1.5)
        img_height = round(text_height * 1.5)
    text_x = (img_width - text_width) // 2
    text_y = (img_height - text_height) // 2
    text_im = draw_text(text_x, text_y, text, fg_color, font, text_border=text_border, text_gradient=text_gradient,
                        text_skew=text_skew, text_rotate=text_rotate, char_spacing=char_spacing,
                        text_shadow=text_shadow, text_width_ratio=text_width_ratio, text_italic=text_italic,
                        text_height_ratio=text_height_ratio, image_width=img_width, image_height=img_height,
                        italic_ratio=italic_ratio, use_text_persp_trans=use_text_persp_trans,
                        text_persp_trans_params=text_persp_trans_params, raise_exception=raise_exception,
                        use_default_render=use_default_render, return_mask=return_mask)
    if return_mask:
        text_im, im_mask = text_im
        # im_mask = np.array(im_mask)
        # for j in range(1, len(text) + 1):
        #     char_mask = (im_mask == j).astype(np.uint8) * 255
        #     Image.fromarray(char_mask, "L").show()
    font = None
    img_width, img_height = text_im.size

    top_row_cnt, bot_row_cnt, left_row_cnt, right_row_cnt = get_image_each_empty_row_nums(text_im)

    text_width = img_width - left_row_cnt - right_row_cnt
    text_height = img_height - top_row_cnt - bot_row_cnt

    paddings = calc_paddings(paddings, text_width, text_height, min_len)

    bg_w = text_width + paddings["left"] + paddings["right"]
    bg_h = text_height + paddings["top"] + paddings["bottom"]
    if bg_w < 1 or bg_h < 1:
        if raise_exception:
            raise Exception("small width or height", bg_w, bg_h, img_width, img_height, top_row_cnt, bot_row_cnt,
                            left_row_cnt, right_row_cnt)
        else:
            bg_w = 100
            bg_h = 50

    text_bg_im = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))
    text_bg_im.paste(text_im, (paddings["left"] - left_row_cnt, paddings["top"] - top_row_cnt))
    text_im.close()
    text_im = None

    if return_mask:
        tmp_im_mask = Image.new("L", (bg_w, bg_h), "black")
        tmp_im_mask.paste(im_mask, (paddings["left"] - left_row_cnt, paddings["top"] - top_row_cnt))
        im_mask = tmp_im_mask

    if use_bg_color:
        if bg_gradient is None:
            bg_img = Image.new("RGBA", (bg_w, bg_h), tuple(bg_color))
        elif bg_gradient["type"] == "linear":
            grad_array = get_multiple_gradation(bg_w, bg_h, bg_gradient)
            bg_img = Image.fromarray(grad_array.astype(np.uint8), mode="RGBA")
    else:
        bg_img = resize_bg_image(bg_img_path, bg_img_scale, bg_img_width_ratio, bg_img_height_ratio, pos_ratio,
                                 use_extend_bg_image, bg_w, bg_h, raise_exception)

    if bg_img.size != text_bg_im.size:
        bg_img = bg_img.resize(text_bg_im.size)
    im = Image.alpha_composite(bg_img, text_bg_im)

    bg_img.close()
    bg_img = None
    text_bg_im.close()
    text_bg_im = None

    if text_blur is not None and text_blur > 0:
        for i in range(text_blur):
            im = im.filter(ImageFilter.BLUR)

    if color_mode != "RGBA":
        im = im.convert(color_mode)

    if use_img_persp_trans:
        im = perspective_transform(im, limit_ratio=0.2, params=img_persp_trans_params,
                                   inter=Image.BICUBIC)
        if return_mask:
            im_mask = perspective_transform(im_mask, limit_ratio=0.2, params=img_persp_trans_params,
                                            inter=Image.NEAREST)

    if use_binarize:
        img = np.array(im)
        img = img[:, :, ::-1].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im = Image.fromarray(img)
        img = None

    if target_width and target_height:
        im = im.resize((target_width, target_height), resample=Image.BILINEAR)
        if return_mask:
            im_mask = im_mask.resize((target_width, target_height), resample=Image.NEAREST)

    if output_path is None:
        if return_numpy:
            result = np.array(im)
            im.close()
            im = None

            if return_mask:
                mask_result = np.array(im_mask)
                im_mask.close()
                im_mask = None
                return result, mask_result
            else:
                return result
        else:
            if return_mask:
                return im, im_mask
            else:
                return im
    else:
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        im.save(output_path, format=image_format, subsampling=0, quality=image_quality)
        im.close()
        im = None
        if return_info:
            return img_width, img_height, text_x, text_y
        else:
            return True
