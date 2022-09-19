from loguru import logger
import datetime
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import get_rank, get_world_size, is_main_process, synchronize

import argparse
import random
import warnings


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = get_rank()
    is_main_node = is_main_process()
    world_size = get_world_size()
    data_loader = exp.get_gen_data_loader(world_size > 1)

    cnt = 0
    results = []
    total_ratio = 0.
    total_area = 0.
    total_time = 0.
    est = time.time()
    max_iter = len(data_loader)
    iter_loader = iter(data_loader)
    for i in range(max_iter):
        data = next(iter_loader)
        total_time += time.time() - est
        data['boxes'] = data['boxes'].squeeze(0)
        cnt += data['boxes'].shape[0]
        w = data['boxes'][:, 2] - data['boxes'][:, 0]
        h = data['boxes'][:, 3] - data['boxes'][:, 1]
        total_ratio += (h / w).sum()
        total_area += (h * w).sum()
        results.append({'image_id': int(data['image_id']),
                       'boxes': data['boxes'].detach().cpu().numpy()})
        if is_main_node:
            if i % 50 == 0:
                eta_seconds = total_time / (i + 1) * (max_iter - i - 1)
                eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info('Progress: [%d/%d] %s, current thread[%d]: %d boxes, avg ratio: %.2f, avg area: %.2f' %
                            (i, max_iter, eta_str, rank, cnt, float(total_ratio) / max(cnt, 1), float(total_area) / max(cnt, 1)))
        if i == 3:
            logger.info('Testing saving...')
            exp.dump_pseu_labels(results, rank, world_size=world_size)
            synchronize()
            if is_main_node:
                exp.collect_pseu_labels(world_size)
            synchronize()
        est = time.time()
    exp.dump_pseu_labels(results, rank, world_size=world_size)
    synchronize()
    if is_main_node:
        exp.collect_pseu_labels(world_size)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )
