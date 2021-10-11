# ---------------------------------------------------------------
# __init__.py
# Set-up time: 2020/4/28 下午8:44
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from .visual_spatial import VisualSpatialExtractor
from .normal import NormalExtractor

__all__ = ['VisualSpatialExtractor', 'NormalExtractor']