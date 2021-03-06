color_joined = [(128, 0, 0, 255), (139, 0, 0, 255), (165, 42, 42, 255), (178, 34, 34, 255),
                (220, 20, 60, 255), (255, 0, 0, 255), (255, 99, 71, 255), (255, 127, 80, 255),
                (205, 92, 92, 255), (240, 128, 128, 255), (233, 150, 122, 255), (250, 128, 114, 255),
                (255, 160, 122, 255), (255, 69, 0, 255), (255, 140, 0, 255), (255, 165, 0, 255),
                (255, 215, 0, 255), (184, 134, 11, 255), (218, 165, 32, 255), (238, 232, 170, 255),
                (189, 183, 107, 255), (240, 230, 140, 255), (128, 128, 0, 255), (255, 255, 0, 255),
                (154, 205, 50, 255), (85, 107, 47, 255), (107, 142, 35, 255), (124, 252, 0, 255),
                (127, 255, 0, 255), (173, 255, 47, 255), (0, 100, 0, 255), (0, 128, 0, 255),
                (34, 139, 34, 255), (0, 255, 0, 255), (50, 205, 50, 255), (144, 238, 144, 255),
                (152, 251, 152, 255), (143, 188, 143, 255), (0, 250, 154, 255), (0, 255, 127, 255),
                (46, 139, 87, 255), (102, 205, 170, 255), (60, 179, 113, 255), (32, 178, 170, 255),
                (47, 79, 79, 255), (0, 128, 128, 255), (0, 139, 139, 255), (0, 255, 255, 255),
                (0, 255, 255, 255), (224, 255, 255, 255), (0, 206, 209, 255), (64, 224, 208, 255),
                (72, 209, 204, 255), (175, 238, 238, 255), (127, 255, 212, 255), (176, 224, 230, 255),
                (95, 158, 160, 255), (70, 130, 180, 255), (100, 149, 237, 255), (0, 191, 255, 255),
                (30, 144, 255, 255), (173, 216, 230, 255), (135, 206, 235, 255), (135, 206, 250, 255),
                (25, 25, 112, 255), (0, 0, 128, 255), (0, 0, 139, 255), (0, 0, 205, 255),
                (0, 0, 255, 255), (65, 105, 225, 255), (138, 43, 226, 255), (75, 0, 130, 255),
                (72, 61, 139, 255), (106, 90, 205, 255), (123, 104, 238, 255), (147, 112, 219, 255),
                (139, 0, 139, 255), (148, 0, 211, 255), (153, 50, 204, 255), (186, 85, 211, 255),
                (128, 0, 128, 255), (216, 191, 216, 255), (221, 160, 221, 255), (238, 130, 238, 255),
                (255, 0, 255, 255), (218, 112, 214, 255), (199, 21, 133, 255), (219, 112, 147, 255),
                (255, 20, 147, 255), (255, 105, 180, 255), (255, 182, 193, 255), (255, 192, 203, 255),
                (250, 235, 215, 255), (245, 245, 220, 255), (255, 228, 196, 255), (255, 235, 205, 255),
                (245, 222, 179, 255), (255, 248, 220, 255), (255, 250, 205, 255), (250, 250, 210, 255),
                (255, 255, 224, 255), (139, 69, 19, 255), (160, 82, 45, 255), (210, 105, 30, 255),
                (205, 133, 63, 255), (244, 164, 96, 255), (222, 184, 135, 255), (210, 180, 140, 255),
                (188, 143, 143, 255), (255, 228, 181, 255), (255, 222, 173, 255), (255, 218, 185, 255),
                (255, 228, 225, 255), (255, 240, 245, 255), (250, 240, 230, 255), (253, 245, 230, 255),
                (255, 239, 213, 255), (255, 245, 238, 255), (245, 255, 250, 255), (112, 128, 144, 255),
                (119, 136, 153, 255), (176, 196, 222, 255), (230, 230, 250, 255), (255, 250, 240, 255),
                (240, 248, 255, 255), (248, 248, 255, 255), (240, 255, 240, 255), (255, 255, 240, 255),
                (240, 255, 255, 255), (255, 250, 250, 255), (0, 0, 0, 255), (105, 105, 105, 255),
                (128, 128, 128, 255), (169, 169, 169, 255), (192, 192, 192, 255), (211, 211, 211, 255),
                (220, 220, 220, 255), (245, 245, 245, 255), (255, 255, 255, 255), ]

paddings = {-0.15: 0.0323, -0.14: 0.0323, -0.13: 0.0323, -0.12: 0.0323, -0.11: 0.0323,
            -0.1: 0.0323, -0.09: 0.0323, -0.08: 0.0323, -0.07: 0.0323, -0.06: 0.0323,
            -0.05: 0.0323, -0.04: 0.0323, -0.03: 0.0323, -0.02: 0.0323, -0.01: 0.0323, 0.: 0.0323,
            0.15: 0.0323, 0.14: 0.0323, 0.13: 0.0323, 0.12: 0.0323, 0.11: 0.0323,
            0.1: 0.0323, 0.09: 0.0323, 0.08: 0.0323, 0.07: 0.0323, 0.06: 0.0323,
            0.05: 0.0323, 0.04: 0.0323, 0.03: 0.0323, 0.02: 0.0323, 0.01: 0.031}


def get_train_params():
    return {
        "paddings": {
            "left": {"range": {"v": [-0.15, 0.35], "p": 1.},
                     "separate": {"v": paddings, "p": 0.}},
            "top": {"range": {"v": [-0.15, 0.4], "p": .6},
                    "separate": {"v": paddings, "p": 0.4}},
            "right": {"range": {"v": [-0.15, 0.35], "p": 1.},
                      "separate": {"v": paddings, "p": 0.}},
            "bottom": {"range": {"v": [-0.15, 0.4], "p": .6},
                       "separate": {"v": paddings, "p": 0.4}},
            # "common": {"range": {"v": [-4, 15], "p": 0.1}, "list": {"v": [20], "p": 0.9}},
        },
        "bg": {
            "color": {
                "disjoined": {
                    "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "a": {"range": {"v": [85, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                    "p": 0.4},
                "joined": {
                    # "list": {"v": color_joined, "p": 0.7},
                    "list": {"v": color_joined, "p": 0.7},
                    "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.3},
                    "p": 0.6},
                "p": 0.35
            },
            "gradient": {
                "direction": {
                    "horizontal": {"p": 0.5,
                                   "angle": {
                                       "separate": {"v": {"left": 0.4, "right": 0.4, "vertical": 0.2}, "p": 0.5},
                                       "list": {"v": ["left", "right", "vertical"], "p": 0.5}}},
                    "vertical": {"p": 0.5},
                },
                "anchors": {
                    "count": {
                        "range": {"v": [2, 5], "p": 0.5},
                        "list": {"v": [2, 3, 4, 5], "p": 0.0},
                        "separate": {"v": {2: 0.5, 3: 0.5}, "p": 0.5}
                    },
                    "pos": {
                        "random": {"p": 0.5},
                        "uniform": {"p": 0.5}
                    },
                    "color": {
                        "disjoined": {
                            "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                            "p": 0.6},
                        "joined": {
                            "list": {"v": color_joined, "p": 0.8},
                            "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.2},
                            "p": 0.4},
                    },
                },
                "p": 0.25
            },
            "image": {
                "p": 0.4,  # 0.5,
                "position": {
                    "left": {"range": {"v": [0., 0.7], "p": 0.5},
                             "list": {"v": [0.2, 0.3, 0.4, 0.5], "p": 0.5}},
                    "top": {"range": {"v": [0., 0.7], "p": 0.5}, "list": {"v": [0.2, 0.3, 0.4, 0.5], "p": 0.5}},
                },
                "scale": {
                    "range": {"v": [0.5, 1.5], "p": 1.},
                    "list": {"v": [0.7, 0.8, 0.9, 1.1, 1.2, 1.3], "p": 0.},
                    "p": 0.4
                },
                "height_ratio": {
                    "range": {"v": [0.75, 1.25], "p": 0.5},
                    "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.5},
                    "p": 0.3
                },
                "width_ratio": {
                    "range": {"v": [0.75, 1.25], "p": 0.5},
                    "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.5},
                    "p": 0.3
                }
            }
        },
        "text": {
            "font_size": {
                "range": {"v": [10, 220], "p": 1.},
                "list": {"v": [20, 30, 40, 50, 60, 70, 80, 90, 100], "p": 0.}
            },
            "border": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.5},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.3},
                        "p": 0.5},
                },
                "width": {
                    "range": {"v": [1, 4], "p": 0.5},
                    "separate": {"v": {1: 0.3, 2: 0.4, 3: 0.3}, "p": 0.5},
                },
                "blur_count": {
                    "range": {"v": [1, 2], "p": 0.5},
                    "list": {"v": [1], "p": 1.},
                    "p": 0.2
                },
                "p": 0.3
            },
            "italic": {"range": {"v": [0.1, 1.0], "p": 1.0},
                       "list": {"v": [1], "p": 0.},
                       "p": 0.1},
            "shadow": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.5},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 1.0}, "p": 0.3},
                        "p": 0.5},
                },
                "width": {
                    "range": {"v": [1, 4], "p": 0.5},
                    "separate": {"v": {1: 0.3, 2: 0.4, 3: 0.3}, "p": 0.5},
                },
                "blur_count": {
                    "range": {"v": [1, 2], "p": 0.5},
                    "list": {"v": [1], "p": 1.},
                    "p": 0.3
                },
                "direction": {
                    "bottom_right": {"p": 0.3},
                    "right": {"p": 0.1},
                    "top_right": {"p": 0.1},
                    "top": {"p": 0.1},
                    "top_left": {"p": 0.1},
                    "left": {"p": 0.1},
                    "bottom_left": {"p": 0.1},
                    "bottom": {"p": 0.1},
                },
                "p": 0.3
            },
            "width_ratio": {
                "range": {"v": [0.75, 1.25], "p": 0.3},
                "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.7},
                "p": 0.1
            },
            "height_ratio": {
                "range": {"v": [0.75, 1.25], "p": 0.3},
                "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.7},
                "p": 0.1
            },
            "rotate": {
                "range": {"v": [-75, 75], "p": .5},
                "list": {"v": list([i - 40 for i in range(80)]), "p": 0.5},
                "p": 0.5
            },
            "blur": {
                "range": {"v": [1, 2], "p": 0.2},
                "list": {"v": [1], "p": 0.8},
                "p": 0.
            },
            "fg": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.7},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 0.3, (0, 0, 0, 255): 0.7}, "p": 0.3},
                        "p": 0.3},
                    "p": 0.7
                },
                "gradient": {
                    "direction": {
                        "horizontal": {"p": 0.5,
                                       "angle": {
                                           "separate": {"v": {"left": 0.4, "right": 0.4, "vertical": 0.2},
                                                        "p": 0.5},
                                           "list": {"v": ["left", "right", "vertical"], "p": 0.5}}},
                        "vertical": {"p": 0.5},
                    },
                    "anchors": {
                        "count": {
                            "range": {"v": [2, 4], "p": 0.5},
                            "list": {"v": [2, 3, 4, 5], "p": 0.5},
                        },
                        "pos": {
                            "random": {"p": 0.5},
                            "uniform": {"p": 0.5}
                        },
                        "color": {
                            "disjoined": {
                                "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                                "p": 0.5},
                            "joined": {
                                "list": {"v": color_joined, "p": 1.0},
                                "p": 0.5},
                        },
                    },
                    "p": 0.3
                },

            },
        }
    }


def get_train_params_text_color():
    return {
        "paddings": {
            "left": {"range": {"v": [-0.15, 0.35], "p": 1.},
                     "separate": {"v": paddings, "p": 0.}},
            "top": {"range": {"v": [-0.15, 0.4], "p": .6},
                    "separate": {"v": paddings, "p": 0.4}},
            "right": {"range": {"v": [-0.15, 0.35], "p": 1.},
                      "separate": {"v": paddings, "p": 0.}},
            "bottom": {"range": {"v": [-0.15, 0.4], "p": .6},
                       "separate": {"v": paddings, "p": 0.4}},
            # "common": {"range": {"v": [-4, 15], "p": 0.1}, "list": {"v": [20], "p": 0.9}},
        },
        "bg": {
            "color": {
                "disjoined": {
                    "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                    "a": {"range": {"v": [85, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                    "p": 0.4},
                "joined": {
                    # "list": {"v": color_joined, "p": 0.7},
                    "list": {"v": color_joined, "p": 0.7},
                    "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.3},
                    "p": 0.6},
                "p": 1.0
            },
            "gradient": {
                "direction": {
                    "horizontal": {"p": 0.5,
                                   "angle": {
                                       "separate": {"v": {"left": 0.4, "right": 0.4, "vertical": 0.2}, "p": 0.5},
                                       "list": {"v": ["left", "right", "vertical"], "p": 0.5}}},
                    "vertical": {"p": 0.5},
                },
                "anchors": {
                    "count": {
                        "range": {"v": [2, 5], "p": 0.5},
                        "list": {"v": [2, 3, 4, 5], "p": 0.0},
                        "separate": {"v": {2: 0.5, 3: 0.5}, "p": 0.5}
                    },
                    "pos": {
                        "random": {"p": 0.5},
                        "uniform": {"p": 0.5}
                    },
                    "color": {
                        "disjoined": {
                            "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                            "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                            "p": 0.6},
                        "joined": {
                            "list": {"v": color_joined, "p": 0.8},
                            "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.2},
                            "p": 0.4},
                    },
                },
                "p": 0.
            },
            "image": {
                "p": 0.,  # 0.5,
                "position": {
                    "left": {"range": {"v": [0., 0.7], "p": 0.5},
                             "list": {"v": [0.2, 0.3, 0.4, 0.5], "p": 0.5}},
                    "top": {"range": {"v": [0., 0.7], "p": 0.5}, "list": {"v": [0.2, 0.3, 0.4, 0.5], "p": 0.5}},
                },
                "scale": {
                    "range": {"v": [0.5, 1.5], "p": 1.},
                    "list": {"v": [0.7, 0.8, 0.9, 1.1, 1.2, 1.3], "p": 0.},
                    "p": 0.4
                },
                "height_ratio": {
                    "range": {"v": [0.75, 1.25], "p": 0.5},
                    "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.5},
                    "p": 0.3
                },
                "width_ratio": {
                    "range": {"v": [0.75, 1.25], "p": 0.5},
                    "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.5},
                    "p": 0.3
                }
            }
        },
        "text": {
            "font_size": {
                "range": {"v": [10, 220], "p": 1.},
                "list": {"v": [20, 30, 40, 50, 60, 70, 80, 90, 100], "p": 0.}
            },
            "border": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.5},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 0.5, (0, 0, 0, 255): 0.5}, "p": 0.3},
                        "p": 0.5},
                },
                "width": {
                    "range": {"v": [1, 4], "p": 0.5},
                    "separate": {"v": {1: 0.3, 2: 0.4, 3: 0.3}, "p": 0.5},
                },
                "blur_count": {
                    "range": {"v": [1, 2], "p": 0.5},
                    "list": {"v": [1], "p": 1.},
                    "p": 0.2
                },
                "p": 0.3
            },
            "italic": {"range": {"v": [0.1, 1.0], "p": 1.0},
                       "list": {"v": [1], "p": 0.},
                       "p": 0.1},
            "shadow": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.5},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 1.0}, "p": 0.3},
                        "p": 0.5},
                },
                "width": {
                    "range": {"v": [1, 4], "p": 0.5},
                    "separate": {"v": {1: 0.3, 2: 0.4, 3: 0.3}, "p": 0.5},
                },
                "blur_count": {
                    "range": {"v": [1, 2], "p": 0.5},
                    "list": {"v": [1], "p": 1.},
                    "p": 0.3
                },
                "direction": {
                    "bottom_right": {"p": 0.3},
                    "right": {"p": 0.1},
                    "top_right": {"p": 0.1},
                    "top": {"p": 0.1},
                    "top_left": {"p": 0.1},
                    "left": {"p": 0.1},
                    "bottom_left": {"p": 0.1},
                    "bottom": {"p": 0.1},
                },
                "p": 0.3
            },
            "width_ratio": {
                "range": {"v": [0.75, 1.25], "p": 0.3},
                "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.7},
                "p": 0.1
            },
            "height_ratio": {
                "range": {"v": [0.75, 1.25], "p": 0.3},
                "list": {"v": [0.8, 0.9, 1.1, 1.2], "p": 0.7},
                "p": 0.1
            },
            "rotate": {
                "range": {"v": [-75, 75], "p": .5},
                "list": {"v": list([i - 40 for i in range(80)]), "p": 0.5},
                "p": 0.5
            },
            "blur": {
                "range": {"v": [1, 2], "p": 0.2},
                "list": {"v": [1], "p": 0.8},
                "p": 0.
            },
            "fg": {
                "color": {
                    "disjoined": {
                        "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                        "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                        "p": 0.7},
                    "joined": {
                        "list": {"v": color_joined, "p": 0.7},
                        "separate": {"v": {(255, 255, 255, 255): 0.3, (0, 0, 0, 255): 0.7}, "p": 0.3},
                        "p": 0.3},
                    "p": 1.
                },
                "gradient": {
                    "direction": {
                        "horizontal": {"p": 0.5,
                                       "angle": {
                                           "separate": {"v": {"left": 0.4, "right": 0.4, "vertical": 0.2},
                                                        "p": 0.5},
                                           "list": {"v": ["left", "right", "vertical"], "p": 0.5}}},
                        "vertical": {"p": 0.5},
                    },
                    "anchors": {
                        "count": {
                            "range": {"v": [2, 4], "p": 0.5},
                            "list": {"v": [2, 3, 4, 5], "p": 0.5},
                        },
                        "pos": {
                            "random": {"p": 0.5},
                            "uniform": {"p": 0.5}
                        },
                        "color": {
                            "disjoined": {
                                "r": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "g": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "b": {"range": {"v": [0, 255], "p": 1.0}, "list": {"v": [20], "p": 0.}},
                                "a": {"range": {"v": [120, 255], "p": 0.5}, "list": {"v": [255], "p": 0.5}},
                                "p": 0.5},
                            "joined": {
                                "list": {"v": color_joined, "p": 1.0},
                                "p": 0.5},
                        },
                    },
                    "p": 0.
                },

            },
        }
    }
