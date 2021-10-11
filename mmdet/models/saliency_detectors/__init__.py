# ---------------------------------------------------------------
# __init__.py
# Set-up time: 2021/5/12 11:01
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from .sal_base import BaseSalineyDetector
from .r3net_saliency_detector import R3NetSaliencyDetector
from .scrn_saliency_detector import SCRNSaliencyDetector

__all__ = ['BaseSalineyDetector', 'R3NetSaliencyDetector', 'SCRNSaliencyDetector']