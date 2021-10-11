# ---------------------------------------------------------------
# __init__.py
# Set-up time: 2021/1/30 17:30
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

from .relational_caption_head import RelationalCaptionHead
from .triplelstm_head import TripleLSTMHead
from .att_base_relcaption_head import AttBaseRelationalCaptionHead
from .xlan_croattn_relcaption_head import XlanCrossAttnRelationalCaptionHead
from .updown_croattn_relcaption_head import UpDownCrossAttnRelationalCaptionHead
from .transformer_croattn_relcaption_head import TransformerCrossAttnRelationalCaptionHead
from .hasg_caption_head import HASGCaptionHead


__all__ = ['RelationalCaptionHead', 'TripleLSTMHead',
           'AttBaseRelationalCaptionHead', 'XlanCrossAttnRelationalCaptionHead',
           'TransformerCrossAttnRelationalCaptionHead', 'HASGCaptionHead']