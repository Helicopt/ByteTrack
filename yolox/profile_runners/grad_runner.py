from loguru import logger
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from yolox.models.id_profiling import SimuModel
from yolox.utils.dist import is_main_process
from senseTk.common import TrackSet, VideoClipReader, Det
from senseTk.functions import drawOnImg
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
        self.iter_num = 500

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
        for f in range(length):
            det_f = data['data'][f]['dets']
            emb_f = data['data'][f]['embs']
            mask = det_f[:, 4] > conf_thres
            det_f = det_f[mask]
            if emb_f is not None:
                emb_f = emb_f[mask]
            fr = data['data'][f]['frame']
            # image_id = data['data'][f]['image_id']
            dets[fr-1] = det_f
            embs[fr-1] = emb_f
            # metas.append({'fr': fr, 'image_id': image_id})
            metas[fr-1] = {'fr': fr, }

        input_data = {
            'dets': dets,
            'embeddings': embs,
            'metas': metas,
        }
        return input_data

    def optimize(self, model, observations):
        data = self.load_data(observations)
        seq_len = data['length']
        # print(seq_len)
        klen = 100
        input_data = self.build_input(data, length=klen)
        input_data = to_device(input_data, device='cuda')
        model.train()
        model = model.cuda()
        instance_ptr = 0
        frame_ptr = 0
        while frame_ptr < klen:
            boxes = model.get_boxes()[:instance_ptr, :, frame_ptr]
            dets = input_data['dets'][frame_ptr]
            # print(frame_ptr, dets.shape)
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
    runner = GradRunner()
    output = runner.run(
        pt_file='/home/toka/code/EOD/data/simu/MOT20-01.pt')
    # out.dump()
