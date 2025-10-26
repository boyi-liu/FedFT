from abc import ABC, abstractmethod

class BaseClient(ABC):
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.server = None
        self.dataset = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def run(self, model):
        pass

    @abstractmethod
    def local_test(self, model):
        pass


class BaseServer(ABC):
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def local_run(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def test_all(self):
        pass