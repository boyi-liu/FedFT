from alg.fedft.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        return super().run(model)

class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()