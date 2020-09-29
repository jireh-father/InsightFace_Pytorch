import random


class TextImageParamParser:
    def __init__(self, params):
        self.params = params

    def get_paddings_param(self):
        paddings = self.params["paddings"]
        if "common" in paddings:
            value = float(self._get_range_or_list_or_separate_value(paddings["common"]))
            result = {"left": value, "top": value, "right": value, "bottom": value}
        else:
            result = {}
            for key in paddings:
                result[key] = float(self._get_range_or_list_or_separate_value(paddings[key]))
        return {"paddings": result}

    def _get_color_value(self, color_params):
        sub_params, sub_key = self._get_sub_param(color_params)
        if sub_key == "joined":
            bg_color = self._get_range_or_list_or_separate_value(sub_params)
        elif sub_key == "disjoined":
            rgb = ["r", "g", "b"]
            bg_color = []
            for color_channel in rgb:
                bg_color.append(int(self._get_range_or_list_or_separate_value(sub_params[color_channel])))
            if "a" in sub_params:
                bg_color.append(int(self._get_range_or_list_or_separate_value(sub_params["a"])))
        else:
            raise Exception("unknown bg color sub key", sub_key, sub_params)
        return bg_color

    def get_bg_param(self):
        params, key = self._get_sub_param(self.params["bg"])
        if key == "color":
            bg_color = self._get_color_value(params)
            return {"bg_color": bg_color, "use_bg_color": True}
        elif key == "gradient":
            direction_params, direction_key = self._get_sub_param(params["direction"])
            if direction_key == "horizontal":
                angle = direction_params["angle"]
                gradient_params = {"direction": direction_key,
                                   "angle": self._get_range_or_list_or_separate_value(angle)}
            elif direction_key == "vertical":
                gradient_params = {"direction": direction_key, "angle": "down"}
            else:
                raise Exception("unknown bg gradient direction key", direction_key, direction_params)

            anchor_cnt = int(self._get_range_or_list_or_separate_value(params["anchors"]["count"]))
            _, pos_key = self._get_sub_param(params["anchors"]["pos"])
            anchor_list = []
            prev_pos = 0.
            for i in range(anchor_cnt):
                sub_color_value = self._get_color_value(params["anchors"]["color"])

                if pos_key == "random":
                    pos = random.uniform(prev_pos, 1.)
                    prev_pos += pos
                elif pos_key == "uniform":
                    if anchor_cnt == 2:
                        pos = float(i)
                    else:
                        pos = 1 / (anchor_cnt - 1) * i
                else:
                    raise Exception("unknown gradient anchors pos key", key)

                anchor_list.append({"color": sub_color_value, "pos": pos})
            gradient_params["type"] = "linear"
            gradient_params["anchors"] = anchor_list
            return {"bg_gradient": gradient_params, "use_bg_color": True}
        elif key == "image":
            left = float(self._get_range_or_list_or_separate_value(params["position"]["left"]))
            top = float(self._get_range_or_list_or_separate_value(params["position"]["top"]))
            scale = float(self._get_range_or_list_or_separate_value(params["scale"]))
            height_ratio = float(self._get_range_or_list_or_separate_value(params["height_ratio"]))
            width_ratio = float(self._get_range_or_list_or_separate_value(params["width_ratio"]))
            return {"pos_ratio": [left, top], "use_bg_color": False, "bg_img_scale": scale,
                    "bg_img_height_ratio": height_ratio,
                    "bg_img_width_ratio": width_ratio}
        else:
            raise Exception("unknown bg key", key, params)

    def _use_param(self, param):
        return param["p"] > random.random()

    def get_text_param(self):
        text_params = self.params["text"]
        font_size = int(self._get_range_or_list_or_separate_value(text_params["font_size"]))

        if self._use_param(text_params["border"]):
            # text_border={"color": "red", "width": 2, "blur_count": 0},
            border_color = self._get_color_value(text_params["border"]["color"])
            border_width = int(self._get_range_or_list_or_separate_value(text_params["border"]["width"]))
            if self._use_param(text_params["border"]["blur_count"]):
                blur_count = int(self._get_range_or_list_or_separate_value(text_params["border"]["blur_count"]))
            else:
                blur_count = 0
            text_border = {"color": border_color, "width": border_width, "blur_count": blur_count}
        else:
            text_border = None

        text_italic = self._use_param(text_params["italic"])
        italic_ratio = None
        if text_italic:
            italic_ratio = float(self._get_range_or_list_or_separate_value(text_params["italic"]))

        if self._use_param(text_params["shadow"]):
            shadow_color = self._get_color_value(text_params["shadow"]["color"])
            shadow_width = int(self._get_range_or_list_or_separate_value(text_params["shadow"]["width"]))
            if self._use_param(text_params["shadow"]["blur_count"]):
                shadow_blur_count = int(self._get_range_or_list_or_separate_value(
                    text_params["shadow"]["blur_count"]))
            else:
                shadow_blur_count = 0
            _, shadow_direction = self._get_sub_param(text_params["shadow"]["direction"])
            text_shadow = {"direction": shadow_direction, "width": shadow_width,
                           "blur_count": shadow_blur_count, "color": shadow_color}
        else:
            text_shadow = None

        if self._use_param(text_params["width_ratio"]):
            text_width_ratio = float(self._get_range_or_list_or_separate_value(text_params["width_ratio"]))
        else:
            text_width_ratio = 1.0

        if self._use_param(text_params["height_ratio"]):
            text_height_ratio = float(self._get_range_or_list_or_separate_value(text_params["height_ratio"]))
        else:
            text_height_ratio = 1.0

        if self._use_param(text_params["rotate"]):
            text_rotate = int(self._get_range_or_list_or_separate_value(text_params["rotate"]))
        else:
            text_rotate = 0

        if self._use_param(text_params["blur"]):
            text_blur = int(self._get_range_or_list_or_separate_value(text_params["blur"]))
        else:
            text_blur = 0

        params, key = self._get_sub_param(text_params["fg"])
        if key == "color":
            fg_color = self._get_color_value(params)
            text_gradient = None
        elif key == "gradient":
            fg_color = (0, 0, 0, 0)
            direction_params, direction_key = self._get_sub_param(params["direction"])
            if direction_key == "horizontal":
                angle = direction_params["angle"]
                gradient_params = {"direction": direction_key,
                                   "angle": self._get_range_or_list_or_separate_value(angle)}
            elif direction_key == "vertical":
                gradient_params = {"direction": direction_key, "angle": "down"}
            else:
                raise Exception("unknown bg gradient direction key", direction_key, direction_params)

            anchor_cnt = int(self._get_range_or_list_or_separate_value(params["anchors"]["count"]))
            anchor_list = []
            prev_pos = 0.
            for i in range(anchor_cnt):
                sub_color_value = self._get_color_value(params["anchors"]["color"])
                _, key = self._get_sub_param(params["anchors"]["pos"])
                if key == "random":
                    pos = random.uniform(prev_pos, 1.)
                    prev_pos += pos
                elif key == "uniform":
                    if anchor_cnt == 2:
                        pos = float(i)
                    else:
                        pos = 1 / (anchor_cnt - 1) * i
                else:
                    raise Exception("unknown gradient anchors pos key", key)

                anchor_list.append({"color": sub_color_value, "pos": pos})
            gradient_params["type"] = "linear"
            gradient_params["anchors"] = anchor_list
            text_gradient = gradient_params
        else:
            raise Exception("unknown bg key", key, params)

        return {"font_size": font_size, "text_border": text_border, "text_italic": text_italic,
                "italic_ratio": italic_ratio,
                "text_shadow": text_shadow,
                "text_width_ratio": text_width_ratio,
                "text_height_ratio": text_height_ratio,
                "text_rotate": text_rotate,
                "text_blur": text_blur,
                "fg_color": fg_color,
                "text_gradient": text_gradient}

    def _get_key_by_prob(self, params):
        keys = list(params.keys())
        keys.sort()
        r = random.random()
        prob = 0.
        for key in keys:
            prob += params[key]
            if prob > r:
                return key
        return key

    def _get_sub_param(self, params):
        keys = list(params.keys())
        keys.sort()
        r = random.random()
        prob = 0.
        for key in keys:
            if key == "p":
                continue
            if "p" not in params[key]:
                continue
            prob += params[key]["p"]
            if prob > r:
                return params[key], key
        return params[key], key

    def _get_range_or_list_or_separate_value(self, params):
        data, data_type = self._get_range_or_list_or_separate(params)
        if data_type == "range":
            if isinstance(data["v"][0], int):
                value = random.randint(data["v"][0], data["v"][1])
            else:
                value = random.uniform(data["v"][0], data["v"][1])
        elif data_type == "list":
            value = random.choice(data["v"])
        elif data_type == "separate":
            value = self._get_key_by_prob(data["v"])
        else:
            raise Exception("unkown type", data_type)
        return value

    def _get_range_or_list_or_separate(self, params):
        if "range" in params and "list" not in params and "separate" not in params:
            return params["range"], "range"
        elif "range" not in params and "list" in params and "separate" not in params:
            return params["list"], "list"
        elif "range" not in params and "list" not in params and "separate" in params:
            return params["separate"], "separate"
        else:
            if "range" not in params:
                params["range"] = {"p": 0.0}
            if "list" not in params:
                params["list"] = {"p": 0.0}
            if "separate" not in params:
                params["separate"] = {"p": 0.0}

            r = random.random()
            if params["range"]["p"] > r:
                return params["range"], "range"
            elif params["list"]["p"] + params["range"]["p"] > r:
                return params["list"], "list"
            elif params["list"]["p"] + params["range"]["p"] + params["separate"]["p"] > r:
                return params["separate"], "separate"
            else:
                raise Exception("unknown value key", params)
