#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    is_main_process,
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    torch_load,
    ckpt_exists,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

import datetime
import os
import time


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        self.id_profile = hasattr(self.exp, 'id_profile') and self.exp.id_profile \
            and hasattr(self.exp, 'switched_pseu') and self.exp.switched_pseu
        self.multi_stage = hasattr(self.exp, 'multi_stage') and self.exp.multi_stage

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        track_ids = targets[:, :, 5]
        targets = targets[:, :, :5]
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        if self.id_profile:
            self.profiling_model, self.profiling_data = self.exp.get_profiling_model()
            if self.start_epoch == 0:
                p_data, epoch = self.resume_profiling(return_epoch=True)
                if p_data is None and epoch is None or epoch != 0:
                    self.id_profiling(epoch=0)
                else:
                    self.profiling_data = p_data
            else:
                self.profiling_data, epoch = self.resume_profiling(return_epoch=True)
                if self.multi_stage:
                    target_epoch = (self.start_epoch // self.exp.eval_interval) * self.exp.eval_interval
                    if target_epoch > epoch:
                        self.id_profiling(epoch=target_epoch)
            self.train_loader.dataset._dataset.set_profile(self.profiling_model, self.profiling_data)

        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        #logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )
        if hasattr(self, 'prefetcher'):
            del self.prefetcher
            time.sleep(3)

    def before_epoch(self):
        if is_main_process():
            logger.info("---> start train epoch{}".format(self.epoch + 1))

        if hasattr(self.exp, 'no_mosaic') and self.exp.no_mosaic:
            if is_main_process():
                logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            
            if is_main_process():
                logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            if is_main_process():
                logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            if not (hasattr(self.exp, 'switched_pseu') and self.exp.switched_pseu):
                self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

        self.save_ckpt(ckpt_name="latest")

        if self.id_profile and self.multi_stage:
            if (self.epoch + 1) % self.exp.eval_interval == 0 and self.exp.eval_interval > 1:
                self.id_profiling(epoch=self.epoch + 1)
                self.train_loader.dataset._dataset.set_profile(self.profiling_model, self.profiling_data)
                if is_main_process():
                    logger.info("init prefetcher, this might take one minute or less...")
                self.prefetcher = DataPrefetcher(self.train_loader)

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0 and is_main_process():
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        resume_ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
        if self.args.resume and ckpt_exists(resume_ckpt_file):
            logger.info("resume training")
            # if self.args.ckpt is None:
            #     ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            # else:
            #     ckpt_file = self.args.ckpt
            ckpt_file = resume_ckpt_file

            ckpt = torch_load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            self.best_ap = ckpt.get("best", self.best_ap)
            logger.info(
                "loaded checkpoint '{}' (epoch {}, best ap {})".format(
                    self.args.resume, self.start_epoch, self.best_ap
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch_load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        #self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "best": self.best_ap,
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def get_profile_partition(self):
        rank = self.rank
        all_ids = sorted(self.profiling_data.keys(), key=lambda x: abs(x - 4) * 2 + int(x > 4))
        n = len(all_ids)
        WS = get_world_size()
        S = (n + WS - 1) // WS
        st = rank * S
        en = min(st + S, n)
        return [all_ids[i] for i in range(st, en)]

    def evaluate_specific(self, todos):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        evaluator = self.exp.get_specific_evaluator(todos, batch_size=self.args.batch_size, is_distributed=False)
        preds = self.exp.eval(
            evalmodel, evaluator, False,
        )
        self.model.train()
        return preds

    def id_profiling(self, epoch):
        todos = self.get_profile_partition()
        self.profile_runner = self.exp.get_profile_runner()
        if len(todos) > 0:
            gt_ssl = self.train_loader.dataset._dataset.pseu_boxes(todos)
            if epoch == 0:
                observation = gt_ssl
            else:
                gt_pred = self.evaluate_specific(todos)
                observation = self.profile_runner.merge(gt_pred, gt_ssl)
            self.model.to('cpu')
            torch.cuda.empty_cache()
            if epoch > 0:
                self.profile_runner.set_mode('hybrid')
            new_data = self.profile_runner.optimize(self.profiling_model, {k: self.profiling_data[k] for k in todos}, observation)
            self.model.to(self.device)
            if self.args.occupy:
                occupy_mem(self.local_rank, mem_ratio=0.5)
            self.save_profiling(new_data, epoch)
        if is_main_process():
            logger.info('waiting other processes to sync')
        synchronize()
        self.profiling_data = self.resume_profiling()

    def resume_profiling(self, return_epoch=False):
        for k in self.profiling_data.keys():
            filename = os.path.join(self.file_name, 'profile_%s' % str(k) + "_ckpt.pth.tar")
            if not ckpt_exists(filename):
                if return_epoch:
                    return None, None
                return None
        epoch = None
        for k in self.profiling_data.keys():
            filename = os.path.join(self.file_name, 'profile_%s' % str(k) + "_ckpt.pth.tar")
            d = torch_load(filename)
            kwargs, _ = self.profiling_data[k]
            saved_kwargs = d['data'][0]
            epoch = d['epoch'] if epoch is None else min(epoch, d['epoch'])
            assert len(set(saved_kwargs.keys()) - set(kwargs.keys())) == 0
            for rk in saved_kwargs:
                assert kwargs[rk] == saved_kwargs[rk] or rk == 'track_num'
            kwargs['track_num'] = saved_kwargs['track_num']
            self.profiling_data[k] = (kwargs, d['data'][1])
        if return_epoch:
            return self.profiling_data, epoch 
        return self.profiling_data

    def save_profiling(self, data, epoch):
        for k, v in data.items():
            pkl_dict = {
                'epoch': epoch,
                'data': v,
                'video_id': k,
            }
            save_checkpoint(pkl_dict, False, self.file_name, 'profile_%s' % str(k))