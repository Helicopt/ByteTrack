from loguru import logger
import numpy as np


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
            first_flag = True
            for frame_id in boxes2[vid]:
                boxes_a = boxes1[vid].get(frame_id, None)
                boxes_b = boxes2[vid][frame_id]
                if boxes_a is None:
                    boxes_a = np.zeros((0, 5))
                boxes_c = self.merger.merge(boxes_a, boxes_b, verbose=first_flag)
                first_flag = False
                ret[vid][frame_id] = boxes_c
        return ret

    def optimize(self, model, todos, observations):
        ret = {}
        for k in todos:
            kwargs, state_dict = todos[k]
            model_inst = model(**kwargs)
            model_inst.load_state_dict(state_dict)
            self.runner.set_video_id(int(k))
            self.runner.set_mode(self.mode)
            new_track_num = self.runner.optimize(model_inst, observations[k])
            kwargs['track_num'] = new_track_num
            new_state_dict = model_inst.state_dict()
            ret[k] = kwargs, {k: v.detach().cpu() for k, v in new_state_dict.items()}
        return ret
