import importlib.util
import sys
import numpy as np
import os

from utils.options import args_parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FedSim:
    def __init__(self, args):
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        os.makedirs(f'./{args.suffix}', exist_ok=True)

        output_path = f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        self.output = open(f'./{output_path}.txt', 'a')

        # === route to algorithm module ===
        if importlib.util.find_spec(f'alg.fedft.{args.alg}'):
            alg_module = importlib.import_module(f'alg.fedft.{args.alg}')
        else:
            alg_module = importlib.import_module(f'alg.fedrag.{args.alg}')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.cn), desc="Loading clients...")]
        self.server = alg_module.Server(args, self.clients)

    def simulate(self):
        acc_list = []
        TEST_GAP = self.args.test_gap

        try:
            for rnd in tqdm(range(0, self.args.rnd), desc='Communication round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    # acc = ret_dict['acc']
                    # acc_list.append(acc)

                    # self.output.write(f'[Round {rnd}] Acc: {acc:.2f} | Time: {self.server.wall_clock_time:.2f}s\n')
                    # self.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 10
            acc_avg = np.mean(acc_list[-avg_count:]).item()
            acc_max = np.max(acc_list).item()

            self.output.write('==========Summary==========\n')
            self.output.write(f'[Total] Acc: {acc_avg:.2f} | Max Acc: {acc_max:.2f}\n')


if __name__ == '__main__':
    FedSim(args=args_parser()).simulate()