from loguru import logger
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torchvision.ops as ops
import torch.optim as optim

import numpy as np
from lap import lapjv


class HungarianCoverage(nn.Module):

    def __init__(self) -> None:
        self.T = 1
        super().__init__()

    def iou(self, a_bboxes, b_bboxes):
        a_bboxes = a_bboxes.unsqueeze(1)
        b_bboxes = b_bboxes.unsqueeze(0)
        intersect = torch.min(a_bboxes[:, :, 2:], b_bboxes[:, :, 2:]) - \
            torch.max(a_bboxes[:, :, :2], b_bboxes[:, :, :2])
        intersect = torch.max(intersect, torch.zeros_like(intersect))
        union = a_bboxes[:, :, 2:] + b_bboxes[:, :, 2:] - intersect
        return intersect / union

    def giou(self, a_bboxes, b_bboxes):
        return ops.generalized_box_iou(a_bboxes, b_bboxes)

    def bbox_overlaps(self, a_bboxes, b_bboxes, mode='giou'):
        if mode == 'giou':
            return self.giou(a_bboxes, b_bboxes)
        elif mode == 'iou':
            return self.iou(a_bboxes, b_bboxes)
        else:
            raise NotImplementedError

    def forward(self, boxes, scores, feats, dets, embeddings, verbose=False):
        cover = 0
        ret = []
        # for i, (det, embedding) in enumerate(zip(dets, embeddings)):
        for i in dets:
            det = dets[i]
            embedding = embeddings[i]
            boxes_i = boxes[:, :, i]
            scores_i = scores[:, :, i]
            det_box_i = det[:, :4]
            det_score_i = det[:, 4].reshape(-1, 1)
            overlaps = self.bbox_overlaps(boxes_i, det_box_i)
            feat_i = feats[:, :, i]
            det_feat_i = embedding
            if det_feat_i is not None:
                cos_sim = feat_i.matmul(det_feat_i.T)
            else:
                cos_sim = overlaps.new_zeros(overlaps.shape)
            # print(cos_sim.shape, cos_sim.max(), cos_sim.min())
            # print(boxes_i)
            if overlaps.shape[1] == 0:
                cover -= scores_i.sum() / self.T
                ret.append(det.new_full((boxes_i.shape[0],), -1))
            else:
                d = overlaps + det_score_i.T + cos_sim * 0.5
                d = d.detach().cpu().numpy()
                cost, x, y = lapjv(-d, extend_cost=True)
                x = torch.from_numpy(x).long().to(det.device)
                # print(x)
                ret.append(x)
                pos_mask = x >= 0
                matched = overlaps[pos_mask] + cos_sim[pos_mask] * 0.5
                mx = torch.gather(matched, 1,
                                  x[pos_mask].reshape(-1, 1))
                if verbose and i == 1:
                    print(mx)
                x_det_scores = det_score_i[x[pos_mask]]
                # weighted_mx = (mx * x_det_scores).sum()
                weighted_mx = mx.sum()
                cover += weighted_mx
                # cover += mx.sum()
                # cover += (1 -
                #           torch.abs(scores_i[pos_mask] - x_det_scores)
                #           ).sum() / self.T
                cover += scores_i[pos_mask].sum() / self.T
                cover -= scores_i[~pos_mask].sum() / self.T
        return cover, ret


class LeastAcceleration(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, xys):
        l = 1
        accel = (xys[:, :, l*2:] + xys[:, :, :-l*2] - xys[:, :, l:-l] * 2)
        a = ((accel ** 2).sum(dim=1)).mean()
        return -a


class LeastVariance(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, dfeats):
        a = ((dfeats ** 2).sum(dim=1)).mean() * 10
        return -a


class SimuModel(nn.Module):

    def __init__(self, track_num, anchor_point, seq_length, dim, feat_anchor_point=None):
        super().__init__()
        self._verbose = False
        self.base = 100
        self.track_num = track_num
        self.anchor_point = anchor_point
        if feat_anchor_point is not None:
            self.feat_anchors = feat_anchor_point
        else:
            self.feat_anchors = anchor_point
        assert seq_length >= anchor_point and seq_length >= self.feat_anchors
        # (anchor_points - 1) * K + anchor_points >= seq_length
        self.seq_length = seq_length
        self._real_length = \
            (seq_length - 2) // (anchor_point - 1) * \
            (anchor_point - 1) + anchor_point
        self._feat_length = \
            (seq_length - 2) // (self.feat_anchors - 1) * \
            (self.feat_anchors - 1) + self.feat_anchors
        self.dim = dim
        self.gen_params()
        self.build_models()

    def self_check(self):
        logger.info('N * T = {} * {}'.format(self.track_num, self._real_length))
        for p, s in self.named_parameters():
            logger.info('{}, {}, {}, {}'.format(p, s.shape, s.requires_grad, s.mean()))

    def build_models(self):
        self.coverage_module = HungarianCoverage()
        self.smoothness_module = LeastAcceleration()
        self.integrity_module = LeastVariance()

    def gen_params(self, append=False):
        if append:
            to_append = self.track_num - self.scores.shape[0]
        else:
            to_append = self.track_num
        param_dict = dict(
            dxys=torch.zeros(to_append, 2, self.anchor_point) + 1,
            xys=torch.randn(to_append, 2, 1),
            dwhs=torch.zeros(to_append, 2, self.anchor_point),
            whs=torch.ones(to_append, 2, 1),
            scores=-torch.ones(to_append, 1, self.anchor_point),
            dfeats=torch.zeros(to_append, self.dim, self.feat_anchors),
            feats=torch.randn(to_append, self.dim, 1),
        )
        for k in param_dict.keys():
            ndata = param_dict[k]
            if append:
                odata = getattr(self, k).data
                setattr(self, k, nn.Parameter(
                    torch.cat([odata, ndata.to(odata.device)], dim=0)))
            else:
                setattr(self, k, nn.Parameter(ndata))
        # self.dxys = nn.Parameter(torch.randn(
        #     self.track_num, 2, self.anchor_point) + 1)
        # self.xys = nn.Parameter(torch.randn(self.track_num, 2, 1))
        # self.dwhs = nn.Parameter(torch.zeros(
        #     self.track_num, 2, self.anchor_point))
        # self.whs = nn.Parameter(torch.ones(self.track_num, 2, 1))
        # self.scores = nn.Parameter(-torch.ones(self.track_num,
        #                                         1, self.anchor_point))
        # self.dfeats = nn.Parameter(torch.zeros(
        #     self.track_num, self.dim, self.feat_anchors))
        # self.feats = nn.Parameter(torch.randn(self.track_num, self.dim, 1))

    def first_M(self, m):
        if m < 0:
            return self.track_num
        else:
            return min(self.track_num, m)

    def update_observations(self, observations, start=0, frame=0):
        dets, embeds = observations
        n = dets.shape[0]
        if start + n > self.track_num:
            self.track_num = start + n
            self.gen_params(append=True)
        xy = dets[:, 0:2].clone()
        xy = torch.clamp(xy, min=1)
        # xy = torch.clamp(xy, min=1)
        wh = dets[:, 2:4] - dets[:, 0:2]
        wh = torch.clamp(wh, min=1)
        # intv = (self._real_length - 1) // (self.anchor_point - 1)
        # frame = round(frame / intv)
        self.xys.data[start:start + n, :, 0] = xy.log()
        self.whs.data[start:start + n, :, 0] = wh.log()
        if embeds is not None:
            self.feats.data[start:start + n, :, 0] = embeds

    def forward(self, inputs, verbose=False, M=-1):
        self._verbose = verbose
        dets = inputs['dets']
        embeddings = inputs['embeddings']
        # if isinstance(dets[0], np.ndarray):
        #     dets = [torch.from_numpy(dets_i).float() for dets_i in dets]
        #     embeddings = [torch.from_numpy(e_i).float() for e_i in embeddings]
        coverage, match_recs = self.coverage(dets, embeddings, M=M)
        smoothness = self.smoothness(M=M)
        integrity = self.integrity(M=M)
        outputs = {
            'metric.coverage': coverage,
            'metric.smoothness': smoothness,
            'metric.integrity': integrity,
        }
        return outputs

    def coverage(self, dets, embeddings, M=-1):
        boxes = self.get_boxes(M, with_score=False)
        feats = self.get_feats(M)
        scores = self.get_scores(M)
        cover, ret = self.coverage_module(
            boxes, scores, feats, dets, embeddings, verbose=self._verbose)
        return cover, ret

    def smoothness(self, M=-1):
        M = self.first_M(M)
        # scores = torch.sigmoid(self.scores).detach()
        xys = self.dxys[:M]  # * self.base
        # print(xys.shape)
        return self.smoothness_module(xys)

    def integrity(self, M=-1):
        M = self.first_M(M)
        dfeats = self.dfeats[:M]
        return self.integrity_module(dfeats)

    def get_boxes(self, M=-1, with_score=True):
        M = self.first_M(M)
        dxy = self.dxys[:M] * self.base
        dxy = F.interpolate(
            dxy, (self._real_length,), align_corners=False, mode='linear')
        xy = torch.exp(self.xys[:M]) + dxy
        dwh = torch.sigmoid(self.dwhs[:M])
        dwh = F.interpolate(
            dwh, (self._real_length,), align_corners=False, mode='linear')
        wh = torch.exp(self.whs[:M]) * (1 + (dwh - 0.5) / 2)
        boxes = torch.cat([xy, xy+wh], dim=1)
        if with_score:
            scores = self.get_scores(M=M)
            boxes = torch.cat([boxes, scores], dim=1)
        return boxes

    def get_feats(self, M=-1):
        M = self.first_M(M)
        feats = self.feats[:M] + self.dfeats[:M]
        feats = F.interpolate(
            feats, (self._feat_length, ), align_corners=False, mode='linear')
        feats = F.normalize(feats, dim=1)
        return feats

    def get_scores(self, M=-1):
        M = self.first_M(M)
        scores = torch.sigmoid(self.scores[:M])
        scores = F.interpolate(
            scores, (self._real_length, ), align_corners=False, mode='linear')
        return scores


def vis(m, tag='init'):
    TMAX = 10

    dwh = m.dwhs.data.clone()
    wh = m.whs.data.clone()
    dxy = m.dxys.data.clone()
    xy = m.xys.data.clone()
    scores = m.scores.data.clone()

    tdwh = torch.zeros((3, 5, 2))
    twh = torch.from_numpy(
        np.array([[[10, 10]], [[10, 10]], [[100, 100]]])).log()
    tdxy = torch.from_numpy(np.array([
        [[10, 10], [11, 11], [12, 12], [13, 13], [14, 14], ],
        [[20, 20], [22, 22], [24, 24], [26, 26], [28, 28], ],
        [[200, 200], [222, 222], [240, 240], [260, 260], [280, 280], ],
    ])) / m.base
    txy = torch.zeros((3, 1, 2)) - 100
    tscores = torch.from_numpy(
        np.array([
            [[0.95], [0.95], [0.95], [0.95], [0.95], ],
            [[0.95], [0.95], [0.95], [0.95], [0.95], ],
            [[0.01], [0.01], [0.95], [0.01], [0.01], ],
        ]))
    tscores = -(1. / tscores - 1).log()

    recs = {}
    X = []
    for t in range(TMAX + 1):
        f = t / TMAX
        n_dwh = dwh * (1 - f) + tdwh * f
        n_wh = wh * (1 - f) + twh * f
        n_dxy = dxy * (1 - f) + tdxy * f
        n_xy = xy * (1 - f) + txy * f
        n_scores = scores * (1 - f) + tscores * f
        m.dwhs.data = n_dwh
        m.whs.data = n_wh
        m.dxys.data = n_dxy
        m.xys.data = n_xy
        m.scores.data = n_scores
        tmp_out = m(inp)
        X.append(f)
        for k in tmp_out:
            if k not in recs:
                recs[k] = []
            recs[k].append(float(tmp_out[k]))
    m.dwhs.data = dwh
    m.whs.data = wh
    m.xys.data = xy
    m.dxys.data = dxy
    m.scores.data = scores
    import matplotlib.pyplot as plt
    # print(recs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, recs['metric.coverage'], label='coverage')
    ax.plot(X, recs['metric.smoothness'], label='smoothness')
    ax.legend()
    fig.show()
    plt.show()
    fig.savefig('curve_%s.png' % tag)


if __name__ == '__main__':
    m = SimuModel(3, 3, 5, 16)
    inp = {
        'dets': [
            np.array([[10, 10, 20, 20, 0.95], [20, 20, 30, 30, 0.95]]),
            # np.zeros((0, 4)),
            np.array([[11, 11, 21, 21, 0.95], [22, 22, 32, 32, 0.95],
                     [222, 222, 322, 322, 0.95]]),
            np.array([[12, 12, 22, 22, 0.95], [24, 24, 34, 34, 0.95]]),
            np.array([[13, 13, 23, 23, 0.95], [26, 26, 36, 36, 0.95]]),
            np.array([[14, 14, 24, 24, 0.95], [28, 28, 38, 38, 0.95]]),
        ],
        'embeddings': [
            np.zeros((2, 16)),
            np.zeros((3, 16)),
            np.zeros((2, 16)),
            np.zeros((2, 16)),
            np.zeros((2, 16)),
        ],
    }

    # vis(m)

    # oup = m(inp)
    # print(oup)
    optimizer = optim.AdamW(m.parameters(), lr=0.01)
    for i in range(m.track_num):
        for j in range(10000):
            optimizer.zero_grad()
            oup = m(inp, verbose=(j % 1000 == 0), M=i + 1)
            loss = -oup['metric.coverage'] * 0.1 - \
                oup['metric.smoothness'] * 10
            loss.backward()
            clip_grad_norm_(m.parameters(), 1)
            optimizer.step()
            if j % 1000 == 0:
                print(oup['metric.coverage'])
                print(oup['metric.smoothness'])
                print(i, j, loss.item())
                print(m.get_boxes().permute(0, 2, 1))
                # vis(m, tag=str(i))
