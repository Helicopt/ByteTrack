from loguru import logger
import numpy as np
from yolox.utils import is_main_process
import time


class Runner:

    def __init__(self, runner, merger) -> None:
        self.runner = runner
        self.merger = merger
        self.mode = 'normal'

    def set_mode(self, mode_new):
        self.mode = mode_new

    def merge(self, boxes1, boxes2):
        logger.info("%s vs %s" % (str(boxes1.keys()), str(boxes2.keys())))
        ret = {}
        for vid in boxes2:
            ret[vid] = {}
            first_flag = is_main_process()
            for frame_id in boxes2[vid]:
                boxes_a = boxes1[vid].get(frame_id, None)
                boxes_b = boxes2[vid][frame_id]
                if boxes_a is None:
                    boxes_a = np.zeros((0, 5))
                boxes_c = self.merger.merge(boxes_a, boxes_b, verbose=first_flag)
                first_flag = False
                ret[vid][frame_id] = boxes_c
        return ret

    def optimize(self, model, todos, observations, saver_func=None, epoch=None):
        ret = {}
        be = time.time()
        tot = len(todos)
        for cur, k in enumerate(todos):
            kwargs, (state_dict, results) = todos[k]
            model_inst = model(**kwargs)
            model_inst.load_state_dict(state_dict)
            self.runner.set_video_id(int(k))
            self.runner.set_mode(self.mode)
            new_track_num, new_results = self.runner.optimize(
                model_inst, observations[k], last_results=results)
            kwargs['track_num'] = new_track_num
            new_state_dict = model_inst.state_dict()
            ret[k] = kwargs, ({k: v.detach().cpu() for k, v in new_state_dict.items()}, new_results)
            if saver_func is not None:
                saver_func({k: ret[k]}, epoch)
            if is_main_process():
                eta = int((time.time() - be) / (cur + 1) * (tot - cur - 1))
                eta_str = '{} d {} hrs {} mins'.format(
                    eta // 86400, eta % 86400 // 3600, eta % 3600 // 60)
                logger.info('ptl progress: {}/{}, remaining time: {}'.format(cur+1, tot, eta_str))
        return ret
