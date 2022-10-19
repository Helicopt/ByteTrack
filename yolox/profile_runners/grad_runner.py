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

    def __init__(self):
        self.iter_num = 300
        self._vid = -1

    def set_video_id(self, vid):
        self._vid = vid

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
                'frame': frame + offset,
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
            'length': len(keys),
            'data': data,
        }
        logger.info('Loaded frames with {} ({} per frame) detections'.format(
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
            logger.info('boxes cnt: %d' % cnt)
            logger.info('%s' % cnts)
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
        tr_rec = [{} for i in range(n)]
        for i in range(length):
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
                tr_rec[a][gt_row[b].uid] = tr_rec[a].get(gt_row[b].uid, 0) + 1
        tr = 0
        for i in range(n):
            tmp = 0
            for uid, c in tr_rec[i].items():
                tmp += c * c
            tmp /= length * length
            tr += tmp
        tr /= max(n, 1)
        return {
            'recall': tp_cnt / max(gt_cnt, 1),
            'precision': tp_cnt / max(pd_cnt, 1),
            'track_rate': tr,
        }

    def optimize(self, model, observations):
        data = self.load_data(observations)
        seq_len = data['length']
        # print(seq_len)
        klen = seq_len - 1
        # klen = 100
        input_data = self.build_input(data, length=klen)
        input_data = to_device(input_data, device='cuda')
        model.train()
        model = model.cuda()
        model.self_check()
        instance_ptr = 0
        frame_ptr = 1
        bef_acc = self.calc_acc(input_data['dets'], length=klen, max_area=240000)
        while frame_ptr < klen:
            if frame_ptr not in input_data['dets']:
                frame_ptr += 1
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
                missing = mxs < 0.4
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
                cv2.imwrite('./cache_imgs/%d_sample.jpg' % (frame_ptr + 1), im)
            model.update_observations(
                (missing_dets, missing_embeds), start=instance_ptr, frame=frame_ptr)
            model.train()
            model = model.cuda()
            model.self_check()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            instance_ptr += len(missing_dets)
            frame_ptr += 2000
            iter_num = self.iter_num if frame_ptr < klen else self.iter_num * 2
            for i in range(iter_num + 1):
                optimizer.zero_grad()
                oup = model(input_data, M=instance_ptr)
                loss = -oup['metric.coverage'] * 0.01 - \
                    oup['metric.smoothness'] * 10 - \
                    oup['metric.integrity'] * 10
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if i % 100 == 0 or frame_ptr == klen and i == self.iter_num - 1:
                    if is_main_process():
                        logger.info('Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
                            i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
                        # print(model.get_boxes())
                        if i % 300 == 0:
                            res = model.get_boxes().permute(
                                0, 2, 1).detach().cpu().numpy()[:instance_ptr]
                            cur_acc = self.calc_acc(res.transpose(1, 0, 2), length=klen, thr=0.3, max_area=240000)
                            logger.info('before acc: %s, current acc: %s' %
                                        (str(bef_acc), str(cur_acc)))
                            continue
                            for j in range(30):
                                img_path = self.restore_path(input_data['metas'], j)
                                row = res[:, j]
                                im = cv2.imread(img_path)
                                for uid, box in enumerate(row):
                                    x1, y1, x2, y2, conf = map(float, box)
                                    D = Det(x1, y1, x2 - x1, y2 -
                                            y1, confidence=conf)
                                    D.uid = uid + 1
                                    if D.area() > 30000:
                                        continue
                                    drawOnImg(im, D, conf=True)
                                os.makedirs('./cache_imgs/iter_{}_{}/'.format(instance_ptr, i),
                                            exist_ok=True)
                                cv2.imwrite(
                                    './cache_imgs/iter_{}_{}/{}.jpg'.format(instance_ptr, i, j + 1), im)
                            for j in range(30):
                                img_path = self.restore_path(input_data['metas'], j)
                                row = input_data['dets'][j]
                                im = cv2.imread(img_path)
                                for uid, box in enumerate(row):
                                    x1, y1, x2, y2, conf = map(float, box)
                                    D = Det(x1, y1, x2 - x1, y2 -
                                            y1, confidence=conf)
                                    D.uid = uid + 1
                                    if D.area() > 30000:
                                        continue
                                    drawOnImg(im, D, conf=True)
                                os.makedirs('./cache_imgs/pseu_{}_{}/'.format(instance_ptr, i),
                                            exist_ok=True)
                                cv2.imwrite(
                                    './cache_imgs/pseu_{}_{}/{}.jpg'.format(instance_ptr, i, j + 1), im)

        # results = model.get_boxes()
        # return results
        return model.track_num


if __name__ == '__main__':
    runner = GradRunner()
    output = runner.run(
        pt_file='/home/toka/code/EOD/data/simu/MOT20-01.pt')
    # out.dump()
