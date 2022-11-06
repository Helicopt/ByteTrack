# encoding: utf-8
import os
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir
from yolox.models.id_profiling import SimuModel

max_epoch_mapping = {
  "su01_mix20": 20,
  "su03_mix20": 20,
  "su12_mix20": 30,
  "su13_mix20": 30,
  "suall_mix20": 60,
}


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.00
        self.width = 1.00
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.trainval_ann = "trainval.json"
        self.val_ann = "val.json"   # change to train.json when running on training set
        self.pseu_prefix = 'pseu_labels'
        self.input_size = (896, 1600)
        self.test_size = (896, 1600)
        # self.test_size = (736, 1920)
        self.random_size = (20, 36)
        self.print_interval = 20
        self.eval_interval = 10
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.basic_lr_per_img = 0.001 / 24.0
        self.warmup_epochs = 1
        self.switched_pseu = False
        self.id_profile = True
        self.ft_tag = "su01_mix20"
        self.is_val = False

    @property
    def ft_tag(self):
        return self._ft_tag

    @ft_tag.setter
    def ft_tag(self, v):
        self._ft_tag = v
        self.max_epoch = max_epoch_mapping[v]
        self.no_aug_epochs = 10

    def get_profiling_model(self) -> Tuple[torch.nn.Module, dict]:
        self.profiling_model = SimuModel
        return self.profiling_model, self.get_profiling_data()

    def get_profiling_data(self) -> dict:
        if not hasattr(self, 'profiling_model'):
            _, d = self.get_profiling_model()
            return d
        self.profiling_data = self.dataset._dataset.get_profiling_data(self.profiling_model)
        return self.profiling_data

    def switch_train_pseu(self):
        self.switched_pseu = True
        # tmp = self.get_data_loader
        # self.get_data_loader = self.get_pseu_data_loader
        # self.get_pseu_data_loader = tmp
        self.max_epoch = 50
        self.eval_interval = 20
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.001 / 24.0
        self.no_aug_epochs = 10

    def get_pseu_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            UnSupMOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = UnSupMOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "suall_mix20"),
            json_file=self.train_ann,
            name='',
            action='load',
            pseu_pkl=os.path.join('YOLOX_outputs', 'yolox_l_gen03_dssl_mix20', self.pseu_prefix + '_ckpt.pth.tar'),
            strategy='profiling',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=600,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        if self.switched_pseu:
            return self.get_pseu_data_loader(batch_size, is_distributed, no_aug=no_aug)
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.ft_tag),
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=600,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            # data_dir=os.path.join(get_yolox_datadir(), "MOT20"),
            data_dir=os.path.join(get_yolox_datadir(), self.ft_tag),
            json_file=self.val_ann if self.is_val else self.trainval_ann,
            img_size=self.test_size,
            name='',  # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_test_real_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MOT20"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='test',  # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def get_specific_loader(self, todos, batch_size, is_distributed, testdev=False):
        from yolox.data import UnSupMOTDataset, ValTransform

        valdataset = UnSupMOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "suall_mix20"),
            json_file=self.train_ann,
            name='',
            action='load',
            strategy='att',
            pseu_pkl=os.path.join('YOLOX_outputs', 'yolox_l_gen03_dssl_mix20',
                                  self.pseu_prefix + '_ckpt.pth.tar'),
            img_size=self.input_size,
            subset=todos,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        )
        # valdataset = COCODataset(
        #     data_dir=None,
        #     json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
        #     name="val2017" if not testdev else "test2017",
        #     img_size=self.test_size,
        #     preproc=ValTransform(
        #         rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        #     ),
        # )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
            batch_size = 8

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_specific_evaluator(self, todos, batch_size, is_distributed, testdev=False):
        from yolox.evaluators.det_evaluator import DetEvaluator

        val_loader = self.get_specific_loader(todos, batch_size, is_distributed, testdev=testdev)
        evaluator = DetEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=0.45,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def get_profile_runner(self):
        from yolox.profile_runners import Runner, GradRunner, NMSMerger, ReplaceMerger
        return Runner(
            runner=GradRunner(
                iter_num=300,
                segment=100,
                gap=10,
                track_lr=0.01,
                track_ratio=0.6,
                track_thr=0.3,
                merge_thr=0.8,
                nms_thr=0.4,
                s_weight=100,
                i_weight=10,
            ),
            merger=ReplaceMerger(

            ),
        )
