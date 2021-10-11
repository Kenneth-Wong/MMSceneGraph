# ---------------------------------------------------------------
# checkpoint.py
# Set-up time: 2020/5/14 下午4:59
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from mmcv.runner import get_dist_info, _load_checkpoint, weights_to_cpu
from collections import OrderedDict
import mmcv
import torch
import os.path as osp
import time

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    load_mapping=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        load_mapping (dict):
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, load_mapping, strict, logger)
    else:
        load_state_dict(model, state_dict, load_mapping, strict, logger)
    return checkpoint


def load_state_dict(module, state_dict, load_mapping=None, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        load_mapping: (dict): Remapping.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # load mapping
    load_mapping_keys = []
    if load_mapping is not None:
        load_mapping_keys = []
        local_name_params = module.named_parameters()
        local_state = {k: v.data for k, v in local_name_params if v is not None}
        align_dict = load_mapping.align_dict
        for module_prefix, state_dict_prefix in align_dict.items():
            for name, param in local_state.items():
                if name.startswith(module_prefix):
                    suffix = name[len(module_prefix):]
                    state_dict_name = state_dict_prefix + suffix
                    param.copy_(state_dict[state_dict_name])
                    load_mapping_keys.append(name)

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if ('num_batches_tracked' not in key) and (key not in load_mapping_keys)
    ]

    if len(load_mapping_keys) > 0:
        err_msg.append('load mapping key in source state_dict: {}\n'.format(
            ', '.join(load_mapping_keys)))

    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def save_checkpoint(model, filename, optimizer=None, scheduler=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    mmcv.mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    torch.save(checkpoint, filename)