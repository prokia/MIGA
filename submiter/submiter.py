from core.runner import get_runner
from submiter.basic_submiter import BasicSubmiter


class Submiter(BasicSubmiter):
    def __init__(self, cfg):
        super(Submiter, self).__init__(cfg)
        self.runner = self.get_runner()(self.cfg, self.writer)

    def get_runner(self):
        runner = get_runner(self.cfg["runner"])
        return runner
