import pickle
import time
import warnings

import numpy as np
import torch
from utils_old import save_checkpoint

from nas_manager import ArchSearchRunManager
import time
from utils_old import *
# from models import *
from run_manager import *
from models.super_nets.super_proxyless import *
from tensorboardX import SummaryWriter
import sys


class CommonwealthMachine:
    def __init__(self, target_hardware=None, config=None, global_run_manager=None,
                 clients_idx_arr=None, clients=None, start_round=0,
                 last_round=None, path='./'):  # 此处config值得商榷, local_client_list值得商榷
        self.hardware = target_hardware
        self.config = config
        self.global_run_manager = copy.deepcopy(global_run_manager)
        self.clients_idx_arr = clients_idx_arr
        self.clients = clients
        self.start_round = start_round
        self.last_round = last_round
        self.local_epoch_number = config.local_epoch_number
        self.path = path
        self._logs_path, self._save_path = None, None
        if self.hardware is not None:
            self.writerTf = SummaryWriter(comment=self.hardware + 'fed_retrain')
        else:
            self.writerTf = SummaryWriter(comment='fed_retrain')
        print('tensorboardX logdir', self.writerTf.logdir)

    def run(self):
        print('self.config.resume: ', self.config.resume)
        if self.config.resume:
            try:
                print('loading global_run_manager model:')
                self.global_run_manager.load_model()
                for id in self.clients_idx_arr:
                    print(id)
                    self.clients[id].load_clients_opt()
            except Exception as e:
                print('Exception about load clients opt:', e)

        self.start_round = self.global_run_manager.round
        best_val_acc = 0
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        for round in range(self.start_round, self.last_round):
            print('round', round)
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            clients_params_arr, clients_data_w = [], []
            round_time = time.time()
            server_model = copy.deepcopy(self.global_run_manager.net.module)
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[idx].train_run_manager(
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    server_model=server_model,
                    writer=self.writerTf)
                clients_params_arr.append(copy.deepcopy(self.clients[idx].return_model_dict()))
                clients_data_w.append(self.clients[idx].get_local_data_weight())
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
            self.writerTf.add_scalar('clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('clients_local_test_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('clients_local_test_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('clients_local_test_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('clients_lr', clients_lr.avg, round)
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('round_time_use', round_time_use, round)
            self.write_log(
                'retrain clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_lr {:.4f}, round_time_use {:.4f}'.format(
                    clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg,
                    clients_lr.avg,
                    round_time_use),
                prefix='retrain')

            # update new fedavg weight
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_run_manager.net.module.load_state_dict(new_weight_fedavg)

            self.global_run_manager.net.module.eval()
            with torch.no_grad():
                # directly use global_run_manager's centralized data for faster inference
                val_loss, val_acc_top1, val_acc_top5 = self.global_run_manager.validate(is_test=False, return_top5=True)
                if best_val_acc < val_acc_top1:
                    best_val_acc = val_acc_top1
                    is_best = True
                else:
                    is_best = False
                # save global model and each client's opt.
                checkpoint = {}
                checkpoint['round'] = round
                checkpoint['state_dict'] = self.global_run_manager.net.module.state_dict()
                for id in self.clients_idx_arr:
                    checkpoint[str(id) + '_weight_optimizer'] = self.clients[
                        id].optimizer.state_dict()
                self.global_run_manager.save_model(checkpoint, is_best=is_best)
                self.writerTf.add_scalar('global_test_loss', val_loss, round)
                self.writerTf.add_scalar('global_test_top1', val_acc_top1, round)
                self.writerTf.add_scalar('global_test_top5', val_acc_top5, round)
        self.writerTf.close()

    def get_server(self):
        return copy.deepcopy(self.global_run_manager)

    def get_model(self):
        return copy.deepcopy(self.global_run_manager.net.module)

    def get_weights(self):
        return copy.deepcopy(self.global_run_manager.net.module.state_dict())

    def test_inference(self):
        # Test inference after completion of training
        val_loss, val_acc_top1, val_acc_top5 = self.global_run_manager.validate(is_test=True, return_top5=True)
        print(val_loss, val_acc_top1, val_acc_top5)

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

def arrange_local_epoch_from_round(global_round=0, local_epoch_number=10):
    return global_round * local_epoch_number, (global_round + 1) * local_epoch_number
