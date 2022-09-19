
class Runner:

    def __init__(self, runner, merger) -> None:
        self.runner = runner
        self.merger = merger

    def merge(self, boxes1, boxes2):
        ret = {}
        for vid in boxes1:
            ret[vid] = {}
            for frame_id in boxes1[vid]:
                boxes_a = boxes1[vid][frame_id]
                boxes_b = boxes2[vid][frame_id]
                boxes_c = self.merger.merge(boxes_a, boxes_b)
                ret[vid][frame_id] = boxes_c
        return ret

    def optimize(self, model, todos, observations):
        ret = {}
        for k in todos:
            kwargs, state_dict = todos[k]
            model_inst = model(**kwargs)
            model_inst.load_state_dict(state_dict)
            self.runner.optimize(model_inst, observations[k])
            new_state_dict = model_inst.state_dict()
            ret[k] = kwargs, {k: v.detach().cpu() for k, v in new_state_dict.items()}
        return ret
