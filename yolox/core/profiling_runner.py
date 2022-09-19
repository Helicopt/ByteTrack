import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from yolox.models.id_profiling import SimuModel
from senseTk.common import TrackSet, VideoClipReader, Det
from senseTk.functions import drawOnImg
import cv2
import os
import pickle as pkl


def to_device(data, device='cuda'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, str):
        return data
    else:
        raise TypeError('Unknown type: {}'.format(type(data)))


class Runner:

    def __init__(self):
        self.iter_num = 500

    @staticmethod
    def load_data(pt_file, embed_file=None):
        with open(pt_file, 'rb') as f:
            data = pkl.load(f)
        if embed_file is not None:
            with open(embed_file, 'rb') as f:
                extra_emb = pkl.load(f)
        else:
            extra_emb = None
        keys = sorted(data.keys())
        tmp = []
        cnt = 0
        for k in keys:
            frame = os.path.splitext(os.path.basename(k))[0]
            d = data[k]
            entry = {
                'frame': frame,
                'image_id': k,
            }
            assert 1 <= len(d) <= 2
            entry['dets'] = d[0]
            cnt += len(d[0])
            if len(d) == 2:
                entry['embs'] = d[1]
            else:
                if isinstance(extra_emb[k], tuple):
                    entry['embs'] = extra_emb[k][-1]
                else:
                    entry['embs'] = extra_emb[k]
            tmp.append(entry)
        data = tmp
        ret = {
            'length': len(keys),
            'data': data,
        }
        print('Loaded {} frames with {} ({} per frame) detections'.format(
            pt_file, cnt, cnt / ret['length']))
        return ret

    def build_input(self, data, length=100, conf_thres=0.6):
        dets = []
        embs = []
        metas = []
        for f in range(length):
            det_f = data['data'][f]['dets']
            emb_f = data['data'][f]['embs']
            mask = det_f[:, 4] > conf_thres
            det_f = det_f[mask]
            emb_f = emb_f[mask]
            fr = data['data'][f]['frame']
            image_id = data['data'][f]['image_id']
            dets.append(det_f)
            embs.append(emb_f)
            metas.append({'fr': fr, 'image_id': image_id})

        input_data = {
            'dets': dets,
            'embeddings': embs,
            'metas': metas,
        }
        return input_data

    def run(self, pt_file, embed_file=None):
        data = self.load_data(pt_file, embed_file)
        seq_len = data['length']
        print(seq_len)
        klen = 100
        model = SimuModel(50, anchor_point=20, seq_length=klen, dim=320)
        input_data = self.build_input(data, length=klen)
        input_data = to_device(input_data, device='cuda')
        model.train()
        model = model.cuda()
        instance_ptr = 0
        frame_ptr = 0
        while frame_ptr < klen:
            boxes = model.get_boxes()[:instance_ptr, :, frame_ptr]
            dets = input_data['dets'][frame_ptr]
            embeds = input_data['embeddings'][frame_ptr]
            if boxes.shape[0] == 0 or dets.shape[0] == 0:
                missing_dets = dets
                missing_embeds = embeds
            else:
                mxs, inds = model.bbox_overlaps(
                    dets[:, :4], boxes[:, :4]).max(dim=1)
                missing = mxs < 0.4
                missing_dets = dets[missing]
                missing_embeds = embeds[missing]
            frame_ptr += 1
            if missing_dets.shape[0] == 0:
                continue
            model.update_observations(
                (missing_dets, missing_embeds), start=instance_ptr, frame=frame_ptr)
            model = model.cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            instance_ptr += len(missing_dets)
            iter_num = self.iter_num if frame_ptr < klen else self.iter_num * 2
            for i in range(iter_num):
                optimizer.zero_grad()
                oup = model(input_data, M=instance_ptr)
                loss = -oup['metric.coverage'] * 0.01 - \
                    oup['metric.smoothness'] * 10 - \
                    oup['metric.integrity'] * 10
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if i % 100 == 0 or frame_ptr == klen and i == self.iter_num - 1:
                    print('Iter: {}, Loss: {} metric.converage: {}, metric.smoothness: {}, metric.integrity: {}'.format(
                        i, loss.item(), oup['metric.coverage'], oup['metric.smoothness'], oup['metric.integrity']))
                    # print(model.get_boxes())
                    if i % 1000 == 0:
                        res = model.get_boxes().permute(
                            0, 2, 1).detach().cpu().numpy()[:instance_ptr]
                        imgs = [x['image_id'] for x in input_data['metas']]
                        for j in range(klen):
                            row = res[:, j]
                            im = imgs[j]
                            im = cv2.imread(im)
                            for uid, box in enumerate(row):
                                x1, y1, x2, y2, conf = map(float, box)
                                D = Det(x1, y1, x2 - x1, y2 -
                                        y1, confidence=conf)
                                D.uid = uid + 1
                                if D.conf < 0.3:
                                    continue
                                drawOnImg(im, D, conf=True)
                            os.makedirs('./iter_{}_{}/'.format(instance_ptr, i),
                                        exist_ok=True)
                            cv2.imwrite(
                                './iter_{}_{}/{}.jpg'.format(instance_ptr, i, j + 1), im)

        results = model.get_boxes()
        return results


if __name__ == '__main__':
    runner = Runner()
    output = runner.run(
        pt_file='/home/toka/code/EOD/data/simu/MOT20-01.pt')
    # out.dump()
