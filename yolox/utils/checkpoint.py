#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch

import io
from petrel_client.client import Client as pc


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def ckpt_exists(filename):
    identifier = 'YOLOX_outputs'
    if identifier in filename:
        pclient = pc()
        s3_path = 's3://toka/%s' % identifier + filename.split(identifier)[-1]
        return pclient.contains(s3_path)
        # with io.BytesIO(pclient.get(s3_path)) as f:
        #     data = torch.load(f, map_location=map_location, **kwargs)
        # return data
    else:
        return os.path.exists(filename)


def torch_load(filename, map_location='cpu', **kwargs):
    identifier = 'YOLOX_outputs'
    if identifier in filename:
        pclient = pc()
        s3_path = 's3://toka/%s' % identifier + filename.split(identifier)[-1]
        with io.BytesIO(pclient.get(s3_path)) as f:
            data = torch.load(f, map_location=map_location, **kwargs)
        return data
    else:
        return torch.load(filename, map_location=map_location, **kwargs)


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    identifier = 'YOLOX_outputs'
    if identifier in save_dir:
        pclient = pc()
        s3_path = 's3://toka/%s' % identifier + save_dir.split(identifier)[-1]
        filename = os.path.join(s3_path, model_name + "_ckpt.pth.tar")
        with io.BytesIO() as f:
            torch.save(state, f)
            pclient.put(filename, f.getvalue())
            if is_best:
                best_filename = os.path.join(s3_path, "best_ckpt.pth.tar")
                pclient.put(best_filename, f.getvalue())
        return
    filename = os.path.join(save_dir, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)
