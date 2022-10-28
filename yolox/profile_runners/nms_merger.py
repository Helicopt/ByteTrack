from loguru import logger
import torch
import torchvision as tv
from torchvision.ops import nms


class NMSMerger:

    def __init__(self):
        self.conf_thr = 0.2
        self.iou_thr = 0.4

    def merge(self, boxes1, boxes2, verbose=False):
        # boxes1 is from pred, boxes2 is from ssl
        boxes1 = boxes1[boxes1[:, 4] > self.conf_thr]
        mask1 = (boxes1[:, 2] - boxes1[:, 0]) > 5
        mask2 = (boxes1[:, 3] - boxes1[:, 1]) > 5
        boxes1 = boxes1[mask1 & mask2]
        boxes1 = torch.from_numpy(boxes1.copy())
        boxes2 = torch.from_numpy(boxes2.copy())
        boxes2[:, 4] = 1e-4
        boxes = torch.cat([boxes1, boxes2])
        ind = nms(boxes[:, :4], boxes[:, 4], iou_threshold=self.iou_thr)
        boxes = boxes[ind]
        if verbose:
            logger.info('merging pred :: %s with ssl :: %s => result :: %s' %
                        (str(boxes1.shape), str(boxes2.shape), str(boxes.shape)))
        return boxes.cpu().numpy()
