from loguru import logger
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from yolox.models.id_profiling import SimuModel
from yolox.utils.dist import is_main_process
from senseTk.common import TrackSet, VideoClipReader, Det
from senseTk.functions import drawOnImg, LAP_Matching
import cv2
import os
import pickle as pkl
from lap import lapjv


def to_device(data, device='cuda'):
    if data is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, (str, slice, int, float)):
        return data
    else:
        raise TypeError('Unknown type: {}'.format(type(data)))


class GradRunner:

    def __init__(self, iter_num=300, segment=100, gap=10, track_lr=0.01, track_ratio=0.6, track_thr=0.3, merge_thr=0.8, nms_thr=0.4, s_weight=100, i_weight=10, ):
        self.iter_num = iter_num
        self.global_iter_num = 10
        self.segment = segment
        self.gap = gap
        self.track_lr = track_lr
        self.track_ratio = track_ratio
        self.track_thr = track_thr
        self.track_merge_thr = merge_thr
        self.post_nms_thr = nms_thr
        self.s_weight = s_weight
        self.i_weight = i_weight

        self._vid = -1
        self._mode = 'normal'

    def set_video_id(self, vid):
        self._vid = vid

    def set_mode(self, mode_new):
        self._mode = mode_new

    @staticmethod
    def load_data(data, embed_data=None):
        # with open(pt_file, 'rb') as f:
        #     data = pkl.load(f)
        # if embed_file is not None:
        #     with open(embed_file, 'rb') as f:
        #         extra_emb = pkl.load(f)
        # else:
        #     extra_emb = None
        keys = sorted(data.keys())
        if keys[0] == 0:
            offset = 1
        else:
            offset = 0
        tmp = []
        cnt = 0
        for frame in keys:
            d = data[frame]
            entry = {
                'frame': frame + offset,  # starts with 1
            }
            if isinstance(d, tuple):
                dets, embeds = d
            else:
                dets = d
                embeds = None
            entry['dets'] = dets
            cnt += len(dets)
            if embeds is not None:
                entry['embs'] = embeds
            else:
                # if isinstance(embed_data[frame], tuple):
                #     entry['embs'] = embed_data[frame][-1]
                # else:
                #     entry['embs'] = embed_data[frame]
                entry['embs'] = None
            tmp.append(entry)
        data = tmp
        ret = {
            'length': max(keys),
            'data': data,
        }
        print('Loaded frames with {} ({} per frame) detections'.format(
            cnt, cnt / ret['length']))
        return ret

    def build_input(self, data, length=100, conf_thres=0.6):
        dets = {}
        embs = {}
        metas = {}
        cnt = 0
        cnts = []
        for f in range(length):
            det_f = data['data'][f]['dets']
            emb_f = data['data'][f]['embs']
            mask = det_f[:, 4] >= 0
            n = int(mask.sum())
            cnt += n
            cnts.append(n)
            det_f = det_f[mask]
            if emb_f is not None:
                emb_f = emb_f[mask]
            fr = data['data'][f]['frame']
            # image_id = data['data'][f]['image_id']
            dets[fr-1] = det_f
            embs[fr-1] = emb_f
            # metas.append({'fr': fr, 'image_id': image_id})
            metas[fr-1] = {'fr': fr, }
        if is_main_process():
            logger.info('%s :: %s' % self.get_seq_info())
            logger.info('boxes cnt: %d' % cnt)
        input_data = {
            'dets': dets,
            'embeddings': embs,
            'metas': metas,
        }
        return input_data

    def get_seq_info(self):
        return {
            1: ('MOT20', 'MOT20-01'),
            2: ('MOT20', 'MOT20-02'),
            3: ('MOT20', 'MOT20-03'),
            4: ('MOT20', 'MOT20-05'),
            6: ('sompt22', 'SOMPT22-02'),
            7: ('sompt22', 'SOMPT22-04'),
            8: ('sompt22', 'SOMPT22-05'),
            9: ('sompt22', 'SOMPT22-07'),
            10: ('sompt22', 'SOMPT22-08'),
            11: ('sompt22', 'SOMPT22-10'),
            12: ('sompt22', 'SOMPT22-11'),
            13: ('sompt22', 'SOMPT22-12'),
            14: ('sompt22', 'SOMPT22-13'),
        }[self._vid]

    def restore_path(self, meta, frame):
        fr = meta[frame]['fr']
        ds, seq = self.get_seq_info()
        img_path = '/mnt/lustre/fengweitao.vendor/code/ByteTrack/datasets/%s/train/%s/img1/%06d.jpg' % (
            ds, seq, fr)
        return img_path

    def calc_acc(self, dets, length, thr=-1., max_area=30000):
        ds, seq = self.get_seq_info()
        gt = TrackSet(
            '/mnt/lustre/fengweitao.vendor/code/ByteTrack/datasets/%s/train/%s/gt/gt.txt' % (ds, seq))
        gt_cnt = 0
        pd_cnt = 0
        tp_cnt = 0
        n = dets[0].shape[0]
        if isinstance(dets, dict):
            for i in dets:
                n = max(n, dets[i].shape[0])
        tr_rec = {}
        if isinstance(length, int):
            iterator = range(length)
        else:
            iterator = range(length[0], length[1] + 1)
        for i in iterator:
            gt_row = gt[i + 1]
            dt_row_ = dets[i]
            thr_mask = dt_row_[:, 4] > thr
            areas = (dt_row_[:, 2] - dt_row_[:, 0]) * (dt_row_[:, 3] - dt_row_[:, 1])
            dt_row_ = dt_row_[thr_mask & (areas <= max_area) & (areas > 100)]
            dt_row = [Det(float(d[0]), float(d[1]), float(d[2]) - float(d[0]),
                          float(d[3]) - float(d[1])) for d in dt_row_]
            ma, l, r = LAP_Matching(dt_row, gt_row, lambda x, y: x.iou(y) if x.iou(y) > 0.4 else 0.)
            tp_cnt += len(ma)
            gt_cnt += len(gt_row)
            pd_cnt += len(dt_row)
            for a, b in ma:
                gid = gt_row[b].uid
                if gid not in tr_rec:
                    tr_rec[gid] = {}
                tr_rec[gid][a] = tr_rec[gid].get(a, 0) + 1
        tr = 0
        tr_cov = 0
        if not isinstance(length, int):
            length = length[1] - length[0] + 1
        for i in tr_rec:
            tmp = 0
            tmp_cov = 0
            for uid, c in tr_rec[i].items():
                tmp += c * c
                tmp_cov += c
            tmp /= length * length
            tmp_cov /= length
            tr += tmp
            tr_cov += tmp_cov
        tr /= max(len(tr_rec), 1)
        tr_cov /= max(len(tr_rec), 1)
        return {
            'recall': tp_cnt / max(gt_cnt, 1),
            'precision': tp_cnt / max(pd_cnt, 1),
            'track_rate': tr,
            'track_coverage': tr_cov,
        }

    def visualize(self, model, input_data, bef_acc, klen, instance_ptr, iter_num=0):
        _, seq = self.get_seq_info()
        res = model.get_boxes(M=instance_ptr, length=klen+1).permute(
            0, 2, 1).detach().cpu().numpy()
        res_merged = model.get_boxes(M=instance_ptr, length=klen+1, merge=True).permute(
            0, 2, 1).detach().cpu().numpy()
        cur_acc = self.calc_acc(res.transpose(
            1, 0, 2), length=klen, thr=0.3, max_area=100000)
        merged_acc = self.calc_acc(res_merged.transpose(
            1, 0, 2), length=klen, thr=0.3, max_area=100000)
        logger.info('before acc: %s, current acc: %s, merged acc: %s' %
                    (str(bef_acc), str(cur_acc), str(merged_acc)))
        # continue
        for j in range(klen):
            if j > 100:
                break
            img_path = self.restore_path(input_data['metas'], j)
            row = res[:, j]
            im = cv2.imread(img_path)
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                if D.area() > 100000:
                    continue
                drawOnImg(im, D, conf=True)
            row = input_data['dets'][j]
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                if D.area() > 100000:
                    continue
                drawOnImg(im, D, color=(10, 150, 150))
            row = res_merged[:, j]
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                if D.area() > 300000:
                    continue
                drawOnImg(im, D, color=(10, 110, 250))
            os.makedirs('./cache_imgs/{}_iter_{}_{}/'.format(seq, instance_ptr, iter_num),
                        exist_ok=True)
            cv2.imwrite(
                './cache_imgs/{}_iter_{}_{}/{}.jpg'.format(seq, instance_ptr, iter_num, j + 1), im)

    def visualize2(self, model, input_data, bef_acc, klen, frange, boxes, tag=''):
        # if tag=='m':
        #     print(boxes[0])
        if isinstance(boxes, list):
            if boxes:
                res = torch.stack(boxes, dim=0).permute(0, 2, 1).detach().cpu().numpy()
            else:
                res = np.zeros((0, klen, 5),)
        else:
            res = boxes.permute(0, 2, 1).detach().cpu().numpy()
        _, seq = self.get_seq_info()
        # res = model.get_boxes(M=instance_ptr, length=klen+1).permute(
        #     0, 2, 1).detach().cpu().numpy()
        res_post = self.post_process(boxes).detach().cpu().numpy()
        cur_acc = self.calc_acc(res.transpose(
            1, 0, 2), length=frange, thr=0.3, max_area=100000)
        post_acc = self.calc_acc(res_post.transpose(
            1, 0, 2), length=frange, thr=0.3, max_area=100000)
        if is_main_process():
            logger.info('current tracks: %d' % (res.shape[0], ))
            logger.info('before acc: %s, current acc: %s, post acc: %s' %
                        (str(bef_acc), str(cur_acc), str(post_acc)))
        return
        # if tag=='m':
        #     print(res[0])
        for j in range(*frange):
            # if j - frange[0] > 50:
            #     break
            # if tag=='m':
            #     print(res[0, j])
            img_path = self.restore_path(input_data['metas'], j)
            row = res[:, j]
            im = cv2.imread(img_path)
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                # if j == 0:
                #     print(str(D))
                if D.area() > 100000 or D.conf < 0.01:
                    continue
                drawOnImg(im, D, conf=True)
            row = input_data['dets'][j]
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                if D.area() > 100000:
                    continue
                drawOnImg(im, D, color=(10, 150, 150))
            row = res_post[:, j]
            for uid, box in enumerate(row):
                x1, y1, x2, y2, conf = map(float, box)
                D = Det(x1, y1, x2 - x1, y2 -
                        y1, confidence=conf)
                D.uid = uid + 1
                if D.area() > 300000 or D.conf < 0.01:
                    continue
                drawOnImg(im, D, color=(10, 110, 250))
            os.makedirs('./cache_imgs/{}_{}kl{}_{}_{}/'.format(seq, tag, klen, frange[0], frange[1]),
                        exist_ok=True)
            cv2.imwrite(
                './cache_imgs/{}_{}kl{}_{}_{}/{}.jpg'.format(seq, tag, klen, frange[0], frange[1], j + 1), im)

    def same_way(self, trj_a, trj_b, thr=0.4):
        pos_a = trj_a[4] > 0.5
        pos_b = trj_b[4] > 0.5
        score_mask = pos_a & pos_b
        union = (pos_a | pos_b).sum()
        a_cx = (trj_a[0] + trj_a[2]) / 2.
        a_cy = (trj_a[1] + trj_a[3]) / 2.
        b_cx = (trj_b[0] + trj_b[2]) / 2.
        b_cy = (trj_b[1] + trj_b[3]) / 2.
        d = ((a_cx - b_cx) ** 2 * 4 + (a_cy - b_cy) ** 2 / 4) ** 0.5
        d = d[score_mask]
        a_s = (trj_a[2:4] - trj_a[0:2]).mean()
        b_s = (trj_b[2:4] - trj_b[0:2]).mean()
        if min(a_s, b_s) / max(a_s, b_s) > 0.5 and d.size(0) > 10 and d.size(0) / (union + 1e-8) > 0.5 and d.max() < (a_s + b_s) * thr:
            logger.info('{} < {}! {}, {}, {}'.format(d.max(), (a_s + b_s) *
                        thr, int(pos_a.sum()), int(pos_b.sum()), d.size(0)))
            return True
        return False

    @torch.no_grad()
    def clean_tracks(self, n=None, length=None, thr=0.3, ratio=0.9):
        if n is None:
            n = self.track_num
        boxes = self.get_boxes(M=n)
        if length is not None:
            boxes = boxes[:, :, :length]
        print(boxes.shape)
        valid_mask = (boxes[:, 4] > thr).float()
        cum_scores = (boxes[:, 4] * valid_mask).sum(dim=1)
        # print(cum_scores.max(), cum_scores.median(), cum_scores.min(), cum_scores.mean())
        # print(cum_scores)
        # print(boxes[:, 4].max(dim=1))
        # pos = cum_scores > 20
        pos = boxes.new_zeros((boxes.size(0), ), dtype=torch.bool)
        _, inds = torch.topk(cum_scores, min(max(int(boxes.size(0) * ratio), 1), boxes.size(0)))
        pos[inds] = True
        reserved_num = int(pos.sum())
        for pk in self.param_keys:
            data = getattr(self, pk).data[:n][pos]
            getattr(self, pk).data[:reserved_num] = data
        return reserved_num

    def post_process(self, track_boxes, to_dict=False):

        boxes = torch.stack(track_boxes).permute(0, 2, 1)
        wh = boxes[:, :, 2:4] - boxes[:, :, 0:2]
        pos = (boxes[:, :, 4] > self.track_thr).float()
        ra = (((wh[:, :, 1] / wh[:, :, 0])) * pos).sum(dim=1) / (pos.sum(dim=1) + 1e-9)
        area = (((wh[:, :, 1] * wh[:, :, 0])) * pos).sum(dim=1) / (pos.sum(dim=1) + 1e-9)
        mask = ((ra < 4) & (ra > 1)) & ((area > 200) & (area < 100000))
        boxes = boxes[mask]
        area = area[mask]
        pos = pos[mask]

        x1 = boxes[:, :, 0]
        y1 = boxes[:, :, 1]
        x2 = boxes[:, :, 2]
        y2 = boxes[:, :, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        weights = torch.abs(area - area.median()) - pos.sum(dim=1)
        order = torch.argsort(weights)  # get boxes with more ious first
        keep = []
        while order.size(0) > 0:
            i = order[0]  # pick maxmum iou box
            keep.append(i)
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.maximum(torch.zeros_like(xx2), xx2 - xx1 + 1)  # maximum width
            h = torch.maximum(torch.zeros_like(yy2), yy2 - yy1 + 1)  # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            pos_ = pos[order[1:]]

            pos_mask = pos[i] * pos_

            ovr = (ovr * pos_mask).sum(dim=1) / (((pos[i] + pos_) > 0).float().sum(dim=1) + 1e-9)

            inds = torch.where(ovr <= self.post_nms_thr)[0]
            order = order[inds + 1]

        # area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).mean(dim=1)
        # with torch.no_grad():
        #     cids = [i for i in range(boxes.shape[0])]
        #     for i in range(boxes.shape[0]):
        #         if area[i] > 120000:
        #             continue
        #         for j in range(i + 1, boxes.shape[0]):
        #             if area[j] > 120000:
        #                 continue
        #             if real_cid(cids, i) != real_cid(cids, j):
        #                 if self.same_way(boxes[i], boxes[j]):
        #                     logger.info('{}({})[area {}] => {}({})[area {}]'.format(
        #                         i, cids[real_cid(cids, i)], area[i], j, real_cid(cids, j), area[j]))
        #                     cids[real_cid(cids, i)] = real_cid(cids, j)
        #     clusters = {}
        #     for i in range(boxes.shape[0]):
        #         clusters.setdefault(real_cid(cids, i), []).append(i)
        #     logger.info('clusters {}'.format(clusters))
        #     mask = boxes.new_ones((boxes.shape[0]), dtype=torch.bool)
        #     pos_masks = []
        #     for i in range(boxes.shape[0]):
        #         pos_masks.append(boxes[i][4] > 0.5)
        #     for cid in clusters:
        #         first = clusters[cid][0]
        #         merged_box = boxes[first].clone()
        #         for i in range(1, len(clusters[cid])):
        #             j = clusters[cid][i]
        #             mask[j] = False
        #             obox = boxes[j]
        #             mask_j = pos_masks[first] & pos_masks[j]
        #             merged_box[0][mask_j] = torch.min(merged_box[0][mask_j], obox[0][mask_j])
        #             merged_box[1][mask_j] = torch.min(merged_box[1][mask_j], obox[1][mask_j])
        #             merged_box[2][mask_j] = torch.max(merged_box[2][mask_j], obox[2][mask_j])
        #             merged_box[3][mask_j] = torch.max(merged_box[3][mask_j], obox[3][mask_j])
        #         logger.info('cid[{}] {} => {}'.format(
        #             cid, boxes[first][:, 0], merged_box[:, 0]))
        #         boxes[first] = merged_box
        #     boxes = boxes[mask]
        if len(keep) == 0:
            logger.info('warn: masking out all boxes {}=>{}'.format(len(track_boxes), 0))
            ret = boxes
        else:
            keep = torch.stack(keep)
            # print(keep)
            ret = boxes[keep]
        if to_dict:
            ret_dict = {}
            ids = torch.arange(ret.shape[0], device=ret.device, dtype=ret.dtype) + 1
            ids = ids.reshape(-1, 1, 1).repeat(1, ret.shape[1], 1)
            id_boxes = torch.cat([ret, ids], dim=2)
            for f in range(ret.shape[1]):
                fr = f
                mask = id_boxes[:, f][:, 4] > self.track_thr
                fr_boxes = id_boxes[:, f][mask]
                ret_dict[fr] = fr_boxes
            return ret_dict
        return ret

    def remove(self, model, data, boxes):
        remain = {k: {} for k in data}
        removed = {k: {} for k in data}
        if boxes:
            boxes = torch.stack(boxes, dim=0)
        else:
            boxes = None
        for f in data['dets']:
            dets_ = data['dets'][f]
            boxes_ = boxes[:, :, f] if boxes is not None else None
            if boxes_ is None or dets_.shape[0] == 0:
                matched = dets_.new_ones((dets_.size(0),), dtype=torch.bool)
            else:
                # print(dets_.shape, boxes_.shape)
                mxs, inds = model.coverage_module.bbox_overlaps(
                    dets_[:, :4], boxes_[:, :4]).max(dim=1)
                matched = mxs > 0.3
            for k in ['dets', 'embeddings']:
                if data[k][f] is not None:
                    remain[k][f] = data[k][f][~matched]
                    removed[k][f] = data[k][f][matched]
                else:
                    remain[k][f] = None
                    removed[k][f] = None
            for k in ['metas']:
                remain[k][f] = data[k][f]
                removed[k][f] = data[k][f]
        return remain, removed

    @torch.no_grad()
    def merge(self, model, mboxes, nboxes, frange, gap):
        if mboxes:
            block = torch.stack(mboxes)
            st = frange[0]
            ed = st + gap
            ablock = block[:, :, st:ed]
            bblock = nboxes[:, :, st:ed]
            ovs = []
            for i in range(gap):
                ov = model.coverage_module.bbox_overlaps(
                    ablock[:, :4, i], bblock[:, :4, i])
                ovs.append(ov)
            ovs = torch.stack(ovs).sum(dim=0)
            ovs[ovs < gap * self.track_merge_thr] = 0
            d = ovs.detach().cpu().numpy()
            cost, x, y = lapjv(-d, extend_cost=True)
            new_ = [True] * nboxes.shape[0]
            for i in range(x.shape[0]):
                if x[i] < 0:
                    continue
                if d[i][int(x[i])] < 1e-6:
                    continue
                mboxes[i] = torch.cat([mboxes[i][:, :frange[0]].detach(),
                                       nboxes[int(x[i])][:, frange[0]:frange[1]+1].detach(),
                                       mboxes[i][:, frange[1]+1:].detach()], dim=1)
                new_[int(x[i])] = False
            for i, flag in enumerate(new_):
                if flag:
                    tmp = nboxes[i].detach()
                    tmp[4, :frange[0]] = 0
                    tmp[4, frange[1]+1:] = 0
                    mboxes.append(tmp)
            if is_main_process():
                logger.info('merged {} added {}'.format(nboxes.shape[0] - sum(new_), sum(new_)))
        else:
            for i in range(nboxes.shape[0]):
                tmp = nboxes[i].detach()
                tmp[4, :frange[0]] = 0
                tmp[4, frange[1]+1:] = 0
                mboxes.append(tmp)
        # print(mboxes[0])

    def optimize_new(self, model, observations, last_results=None):
        data = self.load_data(observations)
        seq_len = data['length']
        seq_len = 300
        # print(seq_len)
        klen = 100  # self.segment
        gap = 10  # self.gap
        input_data = self.build_input(data, length=seq_len)
        input_data = to_device(input_data, device='cuda')
        ori_input = input_data
        model.train()
        model = model.cuda()
        model.self_check()
        # all_boxes = []
        m_boxes = []
        for frame_id in range(0, seq_len, klen - gap):
            frange = (frame_id, min(frame_id + klen - 1, seq_len - 1))
            st, ed = frange
            if frange[0] > 0 and frange[1] - frange[0] + 1 <= gap:
                break
            item_cnts = []
            for j in range(*frange):
                item_cnts.append(len(input_data['dets'].get(j, [])))
            item_cnts = np.array(item_cnts)
            instance_ptr = max(int(item_cnts.mean() * self.track_ratio), 1)
            if is_main_process():
                logger.info('track num stats: max {} min {} mean {} median {}'.format(
                    np.max(item_cnts),
                    np.min(item_cnts),
                    np.mean(item_cnts),
                    np.median(item_cnts),
                ))
            model.resize(instance_ptr, clean=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.track_lr, weight_decay=0.)
            bef_acc = self.calc_acc(input_data['dets'], length=frange, max_area=100000)
            iter_num = self.iter_num
            model.update_observations(
                (input_data['dets'][frame_id][:instance_ptr],
                    input_data['embeddings'][frame_id][:instance_ptr] if input_data['embeddings'][0] is not None else None), start=0, frame=0)
            for i in range(iter_num + 1):
                optimizer.zero_grad()
                # ratio = (i + 1) / (iter_num - self.global_iter_num)
                ratio = 1.
                oup = model(input_data, frange=frange)
                loss = -oup['metric.coverage'] - \
                    oup['metric.smoothness'] * self.s_weight - \
                    oup['metric.integrity'] * self.i_weight
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if i % 100 == 0 or i == iter_num:
                    if is_main_process():
                        logger.info('({})[{}~{}]Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
                            model.track_num, frange[0], frange[1], i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
                        # print(model.get_boxes())
            with torch.no_grad():
                one_track = model.get_boxes()
                self.merge(model, m_boxes, one_track, frange, gap)
                # mask = []
                v_boxes = []
                for i in range(one_track.shape[0]):
                    # keep = True
                    # score = one_track[i, 4, max(ed-gap+1, st):ed+1].mean()
                    # if score < self.track_thr:
                    #     keep = False
                    #     all_boxes.append(one_track[i])
                    # logger.info('added traj with ending score {}'.format(score))
                    v_boxes.append(one_track[i])
                    # mask.append(keep)
                # model.recompose(mask)
                self.visualize2(model, ori_input, bef_acc,
                                klen, frange, v_boxes)
            # if is_main_process():
            #     logger.info('clearning up')
            # input_data, _ = self.remove(model, input_data, added)
        with torch.no_grad():
            # one_track = model.get_boxes()
            # for i in range(one_track.shape[0]):
            #     all_boxes.append(one_track[i])
            all_bef_acc = self.calc_acc(input_data['dets'], length=seq_len, max_area=100000)
            # self.visualize2(model, ori_input, all_bef_acc,
            #                 klen, (0, seq_len - 1), all_boxes)
            self.visualize2(model, ori_input, all_bef_acc,
                            klen, (0, seq_len - 1), m_boxes, tag='m')
            ret = self.post_process(m_boxes, to_dict=True)

        # _, input_data = self.remove(ori_input, all_boxes)
        # instance_ptr = len(all_boxes)
        # model.extend_if_insufficient(instance_ptr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        # iter_num = self.iter_num * 2
        # for i in range(iter_num + 1):
        #     optimizer.zero_grad()
        #     # ratio = (i + 1) / (iter_num - self.global_iter_num)
        #     ratio = 1.
        #     oup = model(input_data, M=instance_ptr)
        #     loss = -oup['metric.coverage'] - \
        #         oup['metric.smoothness'] * 100 - \
        #         oup['metric.integrity'] * 10
        #     loss.backward()
        #     clip_grad_norm_(model.parameters(), 1)
        #     optimizer.step()
        #     if i % 100 == 0 or i == self.iter_num:
        #         if is_main_process():
        #             logger.info('Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
        #                 i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
        # all_boxes2 = model.get_boxes(M=instance_ptr, length=klen+1)
        # self.visualize2(model, input_data, bef_acc, klen, instance_ptr, all_boxes2, iter_num=9999)
        # model.resize(instance_ptr)
        model.resize(1, clean=True)
        return 1, ret

    def optimize(self, model, observations, last_results=None):
        return self.optimize_new(model, observations, last_results=None)
        if self._mode == 'hybrid':
            return self.optimize_hybrid(model, observations, last_results=None)
        data = self.load_data(observations)
        seq_len = data['length']
        # print(seq_len)
        klen = seq_len - 1
        klen = 100
        input_data = self.build_input(data, length=klen)
        input_data = to_device(input_data, device='cuda')
        model.train()
        model = model.cuda()
        model.self_check()
        instance_ptr = 0
        anchor_frames = model.get_anchor_frames()
        aptr = 0
        frame_ptr = anchor_frames[aptr]
        bef_acc = self.calc_acc(input_data['dets'], length=klen, max_area=100000)
        while frame_ptr < klen:
            if frame_ptr not in input_data['dets']:
                aptr += 1
                frame_ptr = anchor_frames[aptr]
                continue
            boxes = model.get_boxes()[:instance_ptr, :, frame_ptr]
            dets = input_data['dets'][frame_ptr]
            embeds = input_data['embeddings'][frame_ptr]
            if boxes.shape[0] == 0 or dets.shape[0] == 0:
                missing_dets = dets
                missing_embeds = embeds
            else:
                mxs, inds = model.coverage_module.bbox_overlaps(
                    dets[:, :4], boxes[:, :4]).max(dim=1)
                missing = mxs < 0.3
                missing_dets = dets[missing]
                if embeds is not None:
                    missing_embeds = embeds[missing]
                else:
                    missing_embeds = None
            if missing_dets.shape[0] == 0:
                continue
            if is_main_process():
                logger.info('%d: add %s' % (frame_ptr, missing_dets.shape))
                img_path = self.restore_path(input_data['metas'], frame_ptr)
                im = cv2.imread(img_path)
                for det in input_data['dets'][frame_ptr]:
                    x1, y1, x2, y2, conf = map(float, det[:5])
                    D = Det(x1, y1, x2 - x1, y2 -
                            y1, confidence=conf)
                    drawOnImg(im, D, conf=True)
                for det in missing_dets:
                    x1, y1, x2, y2, conf = map(float, det[:5])
                    D = Det(x1, y1, x2 - x1, y2 -
                            y1, confidence=conf)
                    drawOnImg(im, D, conf=True, color=(0, 255, 0))
                _, seq = self.get_seq_info()
                cv2.imwrite('./cache_imgs/%s_%d_sample.jpg' % (seq, frame_ptr + 1), im)
            missing_cnt = max(int(missing_dets.shape[0] * 0.3), 1)
            model.update_observations(
                (missing_dets[:missing_cnt], missing_embeds[:missing_cnt] if missing_embeds is not None else None), start=instance_ptr, frame=aptr)
            model.train()
            model = model.cuda()
            model.self_check()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            instance_ptr += len(missing_dets)
            aptr += 1
            prev_frame = frame_ptr
            frame_ptr = anchor_frames[aptr]
            cnt = 0
            skip = 100
            while aptr + 1 < len(anchor_frames):
                aptr += 1
                frame_ptr = anchor_frames[aptr]
                cnt += 1
                if cnt >= skip:
                    break
            iter_num = self.iter_num if frame_ptr < klen else self.iter_num * 2
            for i in range(iter_num + 1):
                optimizer.zero_grad()
                # ratio = (i + 1) / (iter_num - self.global_iter_num)
                ratio = 1.
                L = (frame_ptr - prev_frame) * 2
                en_range = round(L * ratio)
                if i < iter_num - self.global_iter_num:
                    oup = model(input_data, M=instance_ptr, frange=(
                        prev_frame, prev_frame + max(en_range, 1)))
                else:
                    oup = model(input_data, M=instance_ptr)
                loss = -oup['metric.coverage'] - \
                    oup['metric.smoothness'] * 100 - \
                    oup['metric.integrity'] * 10
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if i % 100 == 0 or frame_ptr >= klen and i == self.iter_num:
                    if is_main_process():
                        logger.info('Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
                            i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
                        # print(model.get_boxes())
                        if i % 300 == 0:
                            self.visualize(model, input_data, bef_acc,
                                           klen, instance_ptr, iter_num=i)
            new_instance_ptr = model.clean_tracks(n=instance_ptr, length=klen)
            if frame_ptr >= klen:
                model.train()
                model = model.cuda()
                model.self_check()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                for i in range(iter_num + 1):
                    optimizer.zero_grad()
                    oup = model(input_data, M=instance_ptr)
                    loss = -oup['metric.coverage'] - \
                        oup['metric.smoothness'] * 1 - \
                        oup['metric.integrity'] * 10
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
            if is_main_process():
                logger.info('clean %d out of %d tracks and %d remain' %
                            (instance_ptr - new_instance_ptr, instance_ptr, new_instance_ptr))
            instance_ptr = new_instance_ptr
            res = model.get_boxes(M=instance_ptr, length=klen+1).permute(
                0, 2, 1).detach().cpu().numpy()[:instance_ptr]
            res_merged = model.get_boxes(M=instance_ptr, length=klen+1, merge=True).permute(
                0, 2, 1).detach().cpu().numpy()[:instance_ptr]
            cur_acc = self.calc_acc(res.transpose(
                1, 0, 2), length=klen, thr=0.3, max_area=100000)
            merged_acc = self.calc_acc(res_merged.transpose(
                1, 0, 2), length=klen, thr=0.3, max_area=100000)
            logger.info('before acc: %s, current acc: %s, merged acc: %s' %
                        (str(bef_acc), str(cur_acc), str(merged_acc)))
        # results = model.get_boxes(merge=True)
        # return results
        model.resize(instance_ptr)
        return model.track_num

    def optimize_hybrid(self, model, observations, last_results=None):
        if is_main_process():
            logger.info('optimizing in hybrid mode')
        data = self.load_data(observations)
        seq_len = data['length']
        # print(seq_len)
        klen = seq_len - 1
        klen = 100
        input_data = self.build_input(data, length=klen)
        input_data = to_device(input_data, device='cuda')
        model.train()
        model = model.cuda()
        model.self_check()
        instance_ptr = 0
        anchor_frames = model.get_anchor_frames()
        aptr = 0
        frame_ptr = anchor_frames[aptr]
        bef_acc = self.calc_acc(input_data['dets'], length=klen, max_area=100000)
        while frame_ptr < klen:
            if frame_ptr not in input_data['dets']:
                aptr += 1
                frame_ptr = anchor_frames[aptr]
                continue
            boxes = model.get_boxes()[:instance_ptr, :, frame_ptr]
            dets = input_data['dets'][frame_ptr]
            embeds = input_data['embeddings'][frame_ptr]
            if boxes.shape[0] == 0 or dets.shape[0] == 0:
                missing_dets = dets
                missing_embeds = embeds
            else:
                mxs, inds = model.coverage_module.bbox_overlaps(
                    dets[:, :4], boxes[:, :4]).max(dim=1)
                missing = mxs < 0.3
                missing_dets = dets[missing]
                if embeds is not None:
                    missing_embeds = embeds[missing]
                else:
                    missing_embeds = None
            if missing_dets.shape[0] == 0:
                continue
            if is_main_process():
                logger.info('%d: add %s' % (frame_ptr, missing_dets.shape))
                img_path = self.restore_path(input_data['metas'], frame_ptr)
                im = cv2.imread(img_path)
                for det in input_data['dets'][frame_ptr]:
                    x1, y1, x2, y2, conf = map(float, det[:5])
                    D = Det(x1, y1, x2 - x1, y2 -
                            y1, confidence=conf)
                    drawOnImg(im, D, conf=True)
                for det in missing_dets:
                    x1, y1, x2, y2, conf = map(float, det[:5])
                    D = Det(x1, y1, x2 - x1, y2 -
                            y1, confidence=conf)
                    drawOnImg(im, D, conf=True, color=(0, 255, 0))
                _, seq = self.get_seq_info()
                cv2.imwrite('./cache_imgs/%s_%d_sample.jpg' % (seq, frame_ptr + 1), im)
            missing_cnt = max(int(missing_dets.shape[0] * 0.3), 1)
            model.update_observations(
                (missing_dets[:missing_cnt], missing_embeds[:missing_cnt] if missing_embeds is not None else None), start=instance_ptr, frame=aptr)
            model.train()
            model = model.cuda()
            model.self_check()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            instance_ptr += len(missing_dets)
            aptr += 1
            prev_frame = frame_ptr
            frame_ptr = anchor_frames[aptr]
            cnt = 0
            skip = 100
            while aptr + 1 < len(anchor_frames):
                aptr += 1
                frame_ptr = anchor_frames[aptr]
                cnt += 1
                if cnt >= skip:
                    break
            iter_num = self.iter_num if frame_ptr < klen else self.iter_num * 2
            for i in range(iter_num + 1):
                optimizer.zero_grad()
                # ratio = (i + 1) / (iter_num - self.global_iter_num)
                ratio = 1.
                L = (frame_ptr - prev_frame) * 2
                en_range = round(L * ratio)
                if i < iter_num - self.global_iter_num:
                    oup = model(input_data, M=instance_ptr, frange=(
                        prev_frame, prev_frame + max(en_range, 1)))
                else:
                    oup = model(input_data, M=instance_ptr)
                loss = -oup['metric.coverage'] - \
                    oup['metric.smoothness'] * 100 - \
                    oup['metric.integrity'] * 10
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if i % 100 == 0 or frame_ptr >= klen and i == self.iter_num:
                    if is_main_process():
                        logger.info('Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
                            i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
                        # print(model.get_boxes())
                        if i % 300 == 0:
                            self.visualize(model, input_data, bef_acc,
                                           klen, instance_ptr, iter_num=i)
            new_instance_ptr = model.clean_tracks(n=instance_ptr, length=klen)
            if frame_ptr >= klen:
                model.train()
                model = model.cuda()
                model.self_check()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                for i in range(iter_num + 1):
                    optimizer.zero_grad()
                    oup = model(input_data, M=instance_ptr)
                    loss = -oup['metric.coverage'] - \
                        oup['metric.smoothness'] * 100 - \
                        oup['metric.integrity'] * 10
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
            if is_main_process():
                logger.info('clean %d out of %d tracks and %d remain' %
                            (instance_ptr - new_instance_ptr, instance_ptr, new_instance_ptr))
            instance_ptr = new_instance_ptr
            res = model.get_boxes(M=instance_ptr, length=klen+1).permute(
                0, 2, 1).detach().cpu().numpy()[:instance_ptr]
            res_merged = model.get_boxes(M=instance_ptr, length=klen+1, merge=True).permute(
                0, 2, 1).detach().cpu().numpy()[:instance_ptr]
            cur_acc = self.calc_acc(res.transpose(
                1, 0, 2), length=klen, thr=0.3, max_area=100000)
            merged_acc = self.calc_acc(res_merged.transpose(
                1, 0, 2), length=klen, thr=0.3, max_area=100000)
            logger.info('before acc: %s, current acc: %s, merged acc: %s' %
                        (str(bef_acc), str(cur_acc), str(merged_acc)))
        # results = model.get_boxes()
        # return results
        model.resize(instance_ptr)
        return model.track_num


if __name__ == '__main__':
    runner = GradRunner()
    output = runner.run(
        pt_file='/home/toka/code/EOD/data/simu/MOT20-01.pt')
    # out.dump()
