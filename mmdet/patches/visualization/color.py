# ---------------------------------------------------------------
# color.py
# Set-up time: 2020/4/26 上午8:59
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import numpy as np
import seaborn as sns


def color_palette(*args, **kwargs):
    """Obtain a seaborn palette.

    Args:
        palette: str.
        n_colors (int): Number of colors.
        desat (float): saturation, 0.0 ~ 1.0

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.

    """
    palette = sns.color_palette(*args, **kwargs)

    # transform to bgr and uint8
    new_palette = []
    for color in palette:
        color = (np.array(color) * 255).astype(np.uint8)
        r = color[0]
        color[0] = color[2]
        color[2] = r
        color = tuple([int(c) for c in color])
        new_palette.append(color)
    return new_palette


def float_palette(to_rgb=False, *args, **kwargs):
    """Obtain a seaborn palette.

    Args:
        palette: str.
        n_colors (int): Number of colors.
        desat (float): saturation, 0.0 ~ 1.0

    Returns:
        tuple[int]: A tuple of 3 floats indicating BGR channels.

    """
    palette = sns.color_palette(*args, **kwargs)
    new_palette = []
    for color in palette:
        if not to_rgb:
            color = (color[2], color[1], color[0])
        new_palette.append(color)
    return new_palette





