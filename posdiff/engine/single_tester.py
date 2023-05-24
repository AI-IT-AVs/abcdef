from typing import Dict

import torch
import ipdb
from tqdm import tqdm

from posdiff.engine.base_tester import BaseTester
from posdiff.utils.summary_board import SummaryBoard
from posdiff.utils.timer import Timer
from posdiff.utils.common import get_log_string
from posdiff.utils.torch import release_cuda, to_cuda
import numpy as np


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)

        txt = []
        r_mse, r_mae, t_mse, t_mae = [], [], [], []
 
        for iteration, data_dict in pbar:

            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)

            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()

            result_dict = self.eval_step(self.iteration, data_dict, output_dict)

            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)

            result_dict = release_cuda(result_dict)

            cur_r_mse = result_dict['r_mse']
            cur_r_mae = result_dict['r_mae']
            cur_t_mse = result_dict['t_mse']
            cur_t_mae = result_dict['t_mae']
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
 
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)

        r_mse = np.array(r_mse)
        r_mae = np.array(r_mae)
        t_mse = np.array(t_mse)
        t_mae = np.array(t_mae)
        r_rmse, r_mae, t_rmse, t_mae = np.sqrt(np.mean(r_mse) + 1e-8), np.mean(r_mae), np.sqrt(np.mean(t_mse) + 1e-8), np.mean(t_mae)

        txt.append(f'r_rmse {r_rmse:.6f}')
        txt.append(f'r_mae {r_mae:.6f}')
        txt.append(f't_rmse {t_rmse:.6f}')
        txt.append(f't_mae {t_mae:.6f}')
        txt.append('---------------------------------------------------------')


        with open('result_test_best.txt','w') as f:
            for each_line in txt:
                f.writelines(each_line)
                f.writelines('\n')
