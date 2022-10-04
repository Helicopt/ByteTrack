from loguru import logger
try:
    import mc
    mc_enable = True
    # server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
    # client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
    # mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    logger.info('memcache is enabled')
    # print('memcache is enabled')
except ImportError:
    mc_enable = False
    logger.info('memcache is NOT enabled')
    # print('memcache is NOT enabled')
import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from ...utils import torch_load, save_checkpoint

from cython_bbox import bbox_overlaps


def merge_n_clean(boxes, new_boxes, weight=0.5):
    a_boxes = boxes.astype(np.float)
    b_boxes = new_boxes.astype(np.float)
    a_boxes[:, 2:] += a_boxes[:, :2]
    b_boxes[:, 2:] += b_boxes[:, :2]
    overlaps = bbox_overlaps(a_boxes, b_boxes)
    mxs = np.max(overlaps, axis=1)
    # print(mxs.shape)
    pos = mxs > weight
    pos_boxes = boxes[pos]
    # print(overlaps.shape)
    return pos_boxes


def nms_no_score(boxes, thr=0.4):
    a_boxes = boxes.astype(np.float)
    a_boxes[:, 2:] += a_boxes[:, :2]
    ious = bbox_overlaps(a_boxes, a_boxes)
    n = a_boxes.shape[0]
    v = [True] * n
    for i in range(n):
        if not v[i]:
            continue
        inds = np.nonzero(ious[i] > thr)[0]
        for k in inds:
            if k > i:
                v[int(k)] = False
    return boxes[v]


def hard_code_filter(boxes):
    ra = boxes[:, 3] / boxes[:, 2]
    pos = (ra > 0.2) & (ra < 5)
    area = boxes[:, 3] * boxes[:, 2]
    area_mean = np.median(area)
    pos2 = (area > area_mean * 0.4) & (area < area_mean * 8)
    pos = pos & pos2
    return boxes[pos]


def selective_search(img, h, w, res_size=128):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')

    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    boxes = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=1)
    return boxes


def att_search(img, h, w, adj_imgs, res_size=960, max_group=800, w1=0.5, w2=0.3, unmerged=False):
    scale = res_size / img.shape[1]
    rW = res_size
    rH = int(img.shape[0] * scale)
    img = cv2.resize(img, (rW, rH))
    adj_imgs = [cv2.resize(p, (rW, rH)) for p in adj_imgs]
    img_diffs = [np.abs((img.astype(np.float32) - p.astype(np.float32))) / 255 for p in adj_imgs]
    for i in range(len(img_diffs)):
        p = img_diffs[i]
        m = p > 0.05
        p[m] = 1
        p[~m] = 0
        img_diffs[i] = (p * 255).astype(np.uint8)
    all_boxes = []
    for i, p in enumerate(img_diffs):
        hnd = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        hnd.setBaseImage(p)
        hnd.switchToSelectiveSearchFast()
        boxes = hnd.process().astype('float32')
        all_boxes.append(boxes)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    sbx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sby = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    img_gay = sbx / 2 + sby / 2
    bar = img_gay.max() * 0.1
    img_gay[img_gay > bar] = 255
    img_gay[img_gay <= bar] = 0
    img_gay = img_gay.astype(np.uint8)
    img_gay = np.stack([img_gay, img_gay, img_gay], axis=2)
    hnd = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    hnd.setBaseImage(img_gay)
    hnd.switchToSelectiveSearchFast()
    gay_boxes = hnd.process().astype('float32')

    if unmerged:
        blist = [all_boxes[0][:max_group]]
        if len(all_boxes) > 1:
            blist.append(all_boxes[1][:max_group])
        blist.append(gay_boxes[:max_group])
        boxes2 = np.concatenate(blist, axis=0)
        boxes2 = nms_no_score(boxes2, thr=0.5)
        boxes = hard_code_filter(boxes2)
    else:
        boxes0 = all_boxes[0][:max_group]
        if len(all_boxes) > 1:
            boxes1 = merge_n_clean(boxes0, all_boxes[1][:max_group], weight=w1)
        else:
            boxes1 = boxes0

        boxes2 = merge_n_clean(boxes1, gay_boxes[:max_group], weight=w2)
        # boxes2 = boxes1

        boxes2 = nms_no_score(boxes2)
        boxes = hard_code_filter(boxes2)
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    boxes[..., :4] /= scale
    boxes = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=1)

    return boxes


class UnSupMOTDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        action='normal',
        pseu_pkl='',
        img_size=(608, 1088),
        strategy='att',
        search_size=960,
        max_prop=800,
        subset=None,
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.subset = subset
        self.video_infos = {}
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.strategy = strategy
        self.max_prop = max_prop
        self.mclient = None
        self.action = action
        self.track_num = 100
        self.profiling_density = 15
        self.profile_inited = False
        assert self.action in ['generate', 'load']
        if self.action == 'load':
            self.loaded_pseu_labels = self._load_pseu_annotations(pseu_pkl)
        self.search_size = search_size
        if self.subset is not None:
            self.filter_data()

    def filter_data(self):
        ids_ = []
        anno_ = []
        pseu_ = []
        new_idmapping = {}
        for vid in self.subset:
            for frame_id in self.video_infos[vid]:
                id_ = self.video_infos[vid][frame_id]
                index = self.ids2indices[id_]
                new_idmapping[id_] = len(ids_)
                ids_.append(id_)
                anno_.append(self.annotations[index])
                pseu_.append(self.loaded_pseu_labels[index])
        self.ids = ids_
        self.annotations = anno_
        self.loaded_pseu_labels = pseu_
        self.ids2indices = new_idmapping

    def __len__(self):
        return len(self.ids)

    def _load_pseu_annotations(self, pkl):
        data = torch_load(pkl, map_location='cpu')
        mp = {}
        for one in data:
            one['boxes'][:, 4] -= 1
            image_id = one['image_id']
            mp[image_id] = one['boxes']
        loaded_ids = set(mp.keys())
        all_ids_set = set(self.ids)
        logger.info('loaded id vs anno id: %d - %d, intersect %d' %
                    (len(loaded_ids), len(self.ids), len(loaded_ids & all_ids_set)))
        ret = [mp.get(_ids, None) for _ids in self.ids]
        for oid in all_ids_set - loaded_ids:
            logger.info(str(self.annotations[self.ids2indices[oid]][1]))
        return ret

    def _load_coco_annotations(self):
        self.ids2indices = {_ids: i for i, _ids in enumerate(self.ids)}
        ret = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        for vid in self.video_infos:
            logger.info('loaded video [%d]: %d frames' % (vid, len(self.video_infos[vid])))
        return ret

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        if video_id not in self.video_infos:
            self.video_infos[video_id] = {}
        self.video_infos[video_id][frame_id] = id_
        with_label = im_ann.get("labeled", 'no_label') == 'with_label'
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name, with_label)

    def get_profiling_data(self, model_class):
        state_data = {}
        for vid, frames in self.video_infos.items():
            seq_length = (len(frames) + self.profiling_density -
                          1) // self.profiling_density * self.profiling_density
            kwargs = {
                'track_num': self.track_num,
                'seq_length': seq_length,
                'anchor_point': seq_length // self.profiling_density + 1,
                'dim': 64,
                'feat_anchor_point': None,
            }
            m = model_class(**kwargs)
            m_state = {k: v.detach() for k, v in m.state_dict().items()}
            state_data[vid] = (kwargs, m_state)
        return state_data

    def set_profile(self, model_class, profile_data):
        self.profile_data = profile_data
        self.profile_boxes = {}
        for vid, (kwargs, state) in profile_data.items():
            m = model_class(**kwargs)
            m.load_state_dict(state)
            boxes = m.get_boxes()
            self.profile_boxes[vid] = boxes.detach().cpu().permute(2, 0, 1).numpy()
        self.profile_inited = True

    def load_anno(self, index):
        return self.annotations[index][0]

    def _ensure_memcached(self):
        if self.mclient is None:
            # 首先，指定服务器列表文件和配置文件的读取路径
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            # 然后获取一个mc对象
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
        return

    def get_image_data(self, img_file):
        if mc_enable:
            self._ensure_memcached()
            value = mc.pyvector()
            self.mclient.Get(img_file, value)
            value_buf = mc.ConvertBuffer(value)
            img_array = np.frombuffer(value_buf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_file)
        assert img is not None
        return img

    def get_profiling_boxes(self, video_id, frame_id):
        boxes = self.profile_boxes[video_id]
        frame_boxes = boxes[frame_id - 1]
        mask = frame_boxes[:, 4] > 0.3
        areas = (frame_boxes[:, 2] - frame_boxes[:, 0]) * (frame_boxes[:, 3] - frame_boxes[:, 1])
        mask = mask & (areas <= 30000) & (areas > 100)
        ind = np.nonzero(mask)[0]
        offset = video_id * self.track_num + 1
        frame_boxes = frame_boxes[mask]
        frame_boxes[:, 4] = 0
        frame_boxes = np.concatenate([frame_boxes, (ind + offset).reshape(-1, 1)], axis=1)
        return frame_boxes

    def pseu_boxes(self, todos):
        assert self.action == 'load' and self.strategy == 'profiling', 'pseu labels not loaded'
        ret = {}
        for vid in todos:
            ret[vid] = {}
            for frame_id in sorted(self.video_infos[vid]):
                index = self.ids2indices[self.video_infos[vid][frame_id]]
                boxes = self.loaded_pseu_labels[index]
                if boxes is None:
                    boxes = np.zeros((0, 6), dtype=np.float)
                boxes = boxes[:, :5]
                ret[vid][frame_id] = boxes
        return ret

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name, with_label = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        img = self.get_image_data(img_file)

        if with_label:
            boxes = res.copy()
        elif self.action == 'generate':
            h, w = img.shape[:2]
            # selective search has randomness and without caching, the results are better.
            if self.strategy == 'topk':
                boxes = selective_search(img, h, w, res_size=128)
                boxes = boxes[:self.max_prop]
            elif self.strategy == 'att':
                frame_id = img_info[2]
                video_id = img_info[3]
                adj_imgs = []
                if frame_id + 1 in self.video_infos[video_id]:
                    nxt_id = self.video_infos[video_id][frame_id + 1]
                    _, _, fn, _ = self.annotations[self.ids2indices[nxt_id]]
                    nxt_image = self.get_image_data(os.path.join(
                        self.data_dir, self.name, fn
                    ))
                    adj_imgs.append(nxt_image)
                if frame_id - 1 in self.video_infos[video_id]:
                    nxt_id = self.video_infos[video_id][frame_id - 1]
                    _, _, fn, _ = self.annotations[self.ids2indices[nxt_id]]
                    nxt_image = self.get_image_data(os.path.join(
                        self.data_dir, self.name, fn
                    ))
                    adj_imgs.append(nxt_image)
                boxes = att_search(img, h, w, adj_imgs, res_size=self.search_size)
                img = (img, adj_imgs)
            elif self.strategy == 'att_unmerged':
                frame_id = img_info[2]
                video_id = img_info[3]
                adj_imgs = []
                if frame_id + 1 in self.video_infos[video_id]:
                    nxt_id = self.video_infos[video_id][frame_id + 1]
                    _, _, fn, _ = self.annotations[self.ids2indices[nxt_id]]
                    nxt_image = self.get_image_data(os.path.join(
                        self.data_dir, self.name, fn
                    ))
                    adj_imgs.append(nxt_image)
                if frame_id - 1 in self.video_infos[video_id]:
                    nxt_id = self.video_infos[video_id][frame_id - 1]
                    _, _, fn, _ = self.annotations[self.ids2indices[nxt_id]]
                    nxt_image = self.get_image_data(os.path.join(
                        self.data_dir, self.name, fn
                    ))
                    adj_imgs.append(nxt_image)
                boxes = att_search(img, h, w, adj_imgs, res_size=self.search_size, unmerged=True)
                img = (img, adj_imgs)
            # elif self.strategy == 'mc':
            #     boxes = self.load_from_cache(item, img, h, w)
            #     boxes_indicators = np.where(np.random.binomial(1, p=self.dist2[:len(boxes)]))[0]
            #     boxes = boxes[boxes_indicators]
            # elif self.strategy == "random":
            #     boxes = self.load_from_cache(random.choice(range(self.files)), None, None, None) # relies on cache for now
            #     boxes = boxes[:self.max_prop]
            else:
                raise ValueError("No such strategy")

            if self.strategy != 'profiling':
                boxes = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=1)
        else:
            if self.strategy != 'profiling':
                boxes = self.loaded_pseu_labels[index]
                if boxes is None or boxes.shape[0] < 2:
                    return self.pull_item((index + 1) % len(self.ids))
            else:
                assert self.profile_inited, 'must initiate using set_profile first'
                frame_id = img_info[2]
                video_id = img_info[3]
                boxes = self.get_profiling_boxes(video_id, frame_id)

        return img, boxes, img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None and self.action == 'load':
            flag = False
            if isinstance(img, tuple):
                imgs = img[1]
                img = img[0]
                flag = True
            img, target = self.preproc(img, target, self.input_dim)
            if flag:
                img = (img, imgs)
        img = img[0] if isinstance(img, tuple) else img
        if self.action == 'generate':
            return {
                'image': img,
                'img_info': img_info,
                'boxes': target,
                'image_id': img_id,
            }
        else:
            img = img
        return img, target, img_info, img_id
