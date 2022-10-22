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
import time
import threading
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from ...utils import torch_load, save_checkpoint, quick_test, is_main_process

from cython_bbox import bbox_overlaps


def merge_n_clean(boxes, new_boxes, weight=0.5):
    a_boxes = boxes[:, :4].astype(np.float)
    b_boxes = new_boxes[:, :4].astype(np.float)
    if new_boxes.shape[0] == 0:
        return boxes
    a_boxes[:, 2:] += a_boxes[:, :2]
    b_boxes[:, 2:] += b_boxes[:, :2]
    overlaps = bbox_overlaps(a_boxes, b_boxes)
    mxs = np.max(overlaps, axis=1)
    # print(mxs.shape)
    pos = mxs > weight
    pos_boxes = boxes[pos]
    # print(overlaps.shape)
    return pos_boxes


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_no_score(boxes, thr=0.5):
    if boxes.shape[1] == 5:
        scores = boxes[:, 4]
        boxes = boxes[:, :4]
        a_boxes = boxes.astype(np.float32)
    else:
        n = boxes.shape[0]
        a_boxes = boxes.astype(np.float32)
        areas = a_boxes[:, 2] * a_boxes[:, 3]
        sorted_ind = np.argsort(areas)
        scores_ = np.arange(n) / (n + 1)
        scores = np.zeros(n)
        scores[sorted_ind] = scores_
    a_boxes[:, 2:] += a_boxes[:, :2]

    # ious = bbox_overlaps(a_boxes, a_boxes)
    # v = [True] * n
    # for i in range(n):
    #     if not v[i]:
    #         continue
    #     inds = np.nonzero(ious[i] > thr)[0]
    #     for k in inds:
    #         if k > i:
    #             v[int(k)] = False
    # return boxes[v]

    keep = nms(a_boxes, scores, thr)
    return boxes[keep]


def hard_code_filter(boxes):
    ra = boxes[:, 3] / boxes[:, 2]
    pos = (ra > 0.2) & (ra < 5)
    area = boxes[:, 3] * boxes[:, 2]
    area_mean = np.median(area)
    pos2 = (area > area_mean * 0.2) & (area < area_mean * 8)
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


def detect_boxes(p, engine, max_box=300):
    if engine == 'selective_search':
        hnd = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        hnd.setBaseImage(p)
        hnd.switchToSelectiveSearchFast()
        boxes = hnd.process().astype('float32')
        return boxes
    elif engine == 'edge_box':
        model = './edge_model.gz'
        edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
        rgb_im = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv2.ximgproc.createEdgeBoxes()
        edge_boxes.setMaxBoxes(max_box)
        boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)
        if scores is None:
            boxes = np.ones((1, 4), dtype=np.float32)
            scores = np.zeros((1, 1), dtype=np.float32)
        boxes = np.concatenate([boxes, scores], axis=1)
        return boxes
    else:
        raise NotImplementedError('%s is not recognised' % engine)


def att_search(img, h, w, adj_imgs, res_size=960, max_group=300, w1=0.5, w2=0.3, unmerged=False, box_engine='selective_search', use_sobel=False, rgb=False):
    scale = min(res_size / img.shape[1], 1.0)
    rW = int(img.shape[1] * scale)
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
    if use_sobel:
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
    else:
        if rgb:
            img_gay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_gay = img.copy()
    st = time.time()
    img_diffs.append(img_gay)

    all_boxes = [None] * len(img_diffs)

    def mth_detect(p, i):
        boxes = detect_boxes(p, engine=box_engine)
        all_boxes[i] = boxes
    all_threads = []
    for i, p in enumerate(img_diffs):
        thread = threading.Thread(target=mth_detect, args=(p, i))
        thread.start()
        all_threads.append(thread)
    for thread in all_threads:
        thread.join()
    gay_boxes = all_boxes[-1]
    all_boxes = all_boxes[:-1]

    if unmerged:
        blist = [boxes_[:max_group] for boxes_ in all_boxes]
        blist.append(gay_boxes[:max_group])
        boxes2 = np.concatenate(blist, axis=0)
        boxes2 = nms_no_score(boxes2)
        boxes = hard_code_filter(boxes2)
    else:
        n = len(all_boxes)
        for i in range(n):
            all_boxes[i] = merge_n_clean(gay_boxes, all_boxes[i][:max_group])
        boxes_list = []
        if n == 2:
            boxes_list.extend(all_boxes)
        for i in range(n):
            for j in range(i+1, min(n, i+3)):
                boxes_ = merge_n_clean(all_boxes[i], all_boxes[j])

                boxes_list.append(boxes_)
        boxes2 = np.concatenate(boxes_list, axis=0)
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
        box_engine='selective_search',
        search_size=960,
        max_prop=800,
        max_area=240000,
        subset=None,
        preproc=None,
        skip_test=False,
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
        self.box_engine = box_engine
        assert self.box_engine in ['selective_search', 'edge_box']
        assert self.action in ['generate', 'load']
        if self.action == 'load':
            self.loaded_pseu_labels = self._load_pseu_annotations(pseu_pkl)
        self.search_size = search_size
        if self.subset is not None:
            self.filter_data()
        self.max_area = max_area
        if is_main_process() and not skip_test:
            if self.strategy == 'profiling' and not self.profile_inited:
                pass
            else:
                self.pretest_pseudo_labels()

    def pretest_pseudo_labels(self):
        logger.info('quick test >>>')
        metrics_all = {}
        vid_keys = sorted(list(self.video_infos.keys()))
        metrics_threads = [None] * len(vid_keys)

        def thread_vid_metricc(vid, i):
            samples = []
            gts = []
            cnt = 0
            frs = sorted(list(self.video_infos[vid].keys()))
            frs = frs[:20]
            for fr in frs:
                index = self.ids2indices[self.video_infos[vid][fr]]
                dets = self.pull_item(index, adjust=False)[1]
                samples.append(dets)
                gt_ = self.annotations[index][0]
                gts.append(gt_)
                cnt += gt_.shape[0]

            metrics = quick_test(samples, gts)
            metrics_threads[i] = metrics, cnt
        threads_all = []
        for i, vid in enumerate(vid_keys):
            thread = threading.Thread(target=thread_vid_metricc, args=(vid, i))
            threads_all.append(thread)
            thread.start()
        for thread in threads_all:
            thread.join()
        for i, vid in enumerate(vid_keys):
            metrics, cnt = metrics_threads[i]
            for k in metrics:
                logger.info('video[%d].%s: %.6f' % (vid, k, metrics[k]))
                if k not in metrics_all:
                    metrics_all[k] = []
                metrics_all[k].append((metrics[k], cnt))
        for k in metrics_all:
            val_cum = 0
            cnt_cum = 0
            for val, cnt in metrics_all[k]:
                val_cum += val * cnt
                cnt_cum += cnt
            logger.info('total.%s: %.6f' % (k, val_cum / max(cnt_cum, 1)))
        logger.info('<<< quick test')

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
            m_state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
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
        if is_main_process():
            self.pretest_pseudo_labels()

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
        mask = mask & (areas <= self.max_area) & (areas > 100)
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

    def pull_item(self, index, adjust=True):
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
            elif self.strategy == 'att' or self.strategy == 'att_unmerged':
                frame_id = img_info[2]
                video_id = img_info[3]
                adj_imgs = []
                for d in range(-2, 2):
                    if d >= 0:
                        d += 1
                    if (frame_id + d) in self.video_infos[video_id]:
                        nxt_id = self.video_infos[video_id][frame_id + d]
                        _, _, fn, _ = self.annotations[self.ids2indices[nxt_id]]
                        nxt_image = self.get_image_data(os.path.join(
                            self.data_dir, self.name, fn
                        ))
                        adj_imgs.append(nxt_image)
                if len(adj_imgs) == 1:
                    logger.info('video[%d] frame %d is isolated.' % (video_id, frame_id))
                boxes = att_search(img, h, w, adj_imgs, res_size=self.search_size, unmerged=self.strategy.endswith(
                    '_unmerged'), box_engine=self.box_engine)
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
                    if adjust:
                        return self.pull_item((index + 1) % len(self.ids))
                    else:
                        boxes = np.zeros((0, 6), dtype=np.float32)
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
