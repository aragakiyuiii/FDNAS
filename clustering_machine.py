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


class ClusteringMachine:
    def __init__(self, target_hardware=None, config=None, global_server=None,
                 clients_idx_arr=None, clients=None, start_round=0,
                 last_round=None, path='./'):
        self.hardware = target_hardware
        self.config = config
        self.global_server = copy.deepcopy(global_server)
        self.clients_idx_arr = clients_idx_arr
        self.clients = clients
        self.start_round = start_round
        self.last_round = last_round
        self.local_epoch_number = config.local_epoch_number
        self.path = path
        self._logs_path, self._save_path = None, None
        if self.hardware is not None:
            self.writerTf = SummaryWriter(comment=self.hardware + 'fed_search')
        else:
            self.writerTf = SummaryWriter(comment='fed_search')
        print('tensorboardX logdir', self.writerTf.logdir)

    def train_clients(self):
        self.start_round = self.global_server.round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        best_val_acc = 0
        for round in range(self.start_round, self.last_round):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_entropy, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            server_model = copy.deepcopy(self.global_server.net)
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, entropy, lr = self.clients[
                    idx].train(
                    server_model=server_model,
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    writer=self.writerTf)
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(self.clients[idx].get_local_data_weight())
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_entropy.update(entropy)
                clients_lr.update(lr)
            self.writerTf.add_scalar('clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('clients_val_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('clients_val_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('clients_val_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('clients_entropy', clients_entropy.avg, round)
            self.writerTf.add_scalar('clients_lr', clients_lr.avg, round)
            self.write_log(
                'search clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_entropy {:.4f}, clients_lr {:.4f}'.format(
                    clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg, clients_entropy.avg,
                    clients_lr.avg),
                prefix='search')
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('round_time_use', round_time_use, round)

            # update new_fedavg_weight
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.net.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.net.load_state_dict(server_dict)
            self.global_server.write_log('-' * 30 + 'Current Architecture [%d]' % (round + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.global_server.net.blocks):
                self.global_server.write_log('%d. %s' % (idx, block.module_str), prefix='arch')

            # Calculate avg training accuracy over all users at every round
            self.global_server.net.eval()
            with torch.no_grad():
                # directly use global_server's centralized data for faster inference
                val_loss, val_acc_top1, val_acc_top5 = self.global_server.inference(is_test=False)
                if best_val_acc < val_acc_top1:
                    best_val_acc = val_acc_top1
                    is_best = True
                else:
                    is_best = False
                # save global model and each client's opt.
                checkpoint = {}
                checkpoint['round'] = round
                checkpoint['warmup'] = False
                checkpoint['state_dict'] = self.global_server.net.state_dict()
                for id in self.clients_idx_arr:
                    checkpoint[str(id) + '_weight_optimizer'] = self.clients[
                        id].run_manager.optimizer.state_dict()
                    checkpoint[str(id) + '_arch_optimizer'] = self.clients[
                        id].arch_optimizer.state_dict()
                self.global_server.run_manager.save_model(checkpoint, is_best=is_best)
        self.writerTf.close()

    def warmup_clients(self):
        self.warmup_round = self.global_server.warmup_round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))

        for round in range(self.warmup_round, self.config.warmup_n_rounds):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            server_model = copy.deepcopy(self.global_server.net)
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[
                    idx].warm_up(server_model=server_model, start_local_epoch=start_local_epoch,
                                 last_local_epoch=last_local_epoch, writer=self.writerTf)
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(self.clients[idx].get_local_data_weight())
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
            self.writerTf.add_scalar('warmup clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('warmup clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('warmup clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('warmup clients_val_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('warmup clients_val_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('warmup clients_val_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('warmup clients_lr', clients_lr.avg, round)
            self.write_log(
                'warmup clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_lr {:.4f}'.format(clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg,clients_lr.avg),
                prefix='warmup')
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('warmup round_time_use', round_time_use, round)
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.net.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.net.load_state_dict(server_dict)
            # save global model and each client's opt.
            checkpoint = {}
            checkpoint['warmup_round'] = round
            checkpoint['warmup'] = True
            checkpoint['state_dict'] = self.global_server.net.state_dict()
            for id in self.clients_idx_arr:
                checkpoint[str(id) + '_weight_optimizer'] = self.clients[
                    id].run_manager.optimizer.state_dict()
                checkpoint[str(id) + '_arch_optimizer'] = self.clients[
                    id].arch_optimizer.state_dict()
            self.global_server.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')
        checkpoint = {}
        checkpoint['warmup_round'] = self.config.warmup_n_rounds
        checkpoint['warmup'] = False
        checkpoint['state_dict'] = self.global_server.net.state_dict()
        for id in self.clients_idx_arr:
            checkpoint[str(id) + '_weight_optimizer'] = self.clients[
                id].run_manager.optimizer.state_dict()
            checkpoint[str(id) + '_arch_optimizer'] = self.clients[
                id].arch_optimizer.state_dict()
        self.global_server.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')
        self.writerTf.close()

    def run(self):
        if self.config.resume:
            try:
                self.global_server.load_model()
                for id in self.clients_idx_arr:
                    self.clients[id].load_clients_opt()
            except Exception as e:
                print('Exception about load clients opt:', e)
        if self.global_server.warmup:
            self.warmup_clients()
        self.train_clients()

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

    def get_server(self):
        return copy.deepcopy(self.global_server)

    def transfer_to_normal_net(self):
        server_model = copy.deepcopy(self.global_server.net)
        if isinstance(server_model, nn.DataParallel):
            server_model = server_model.module

        normal_net = server_model.cpu().convert_to_normal_net()
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
        os.makedirs(os.path.join(self.global_server.run_manager.path, self.hardware + '_learned_net'), exist_ok=True)
        torch.save(
            {'state_dict': normal_net.state_dict()},
            os.path.join(self.global_server.run_manager.path, self.hardware + '_learned_net/init.pth.tar')
        )

    def test_inference(self):
        # Test inference after completion of training
        AvgTestLoss, TestAccuracyTop1, TestAccuracyTop5 = self.global_server.inference(server_model=self.global_server.net,
                                                                                      is_test=True,
                                                                                      return_top5=True)
        print("|---- Avg Test Loss: {:.2f}%".format(100 * AvgTestLoss))
        print("|---- Test Accuracy Top1: {:.2f}%".format(100 * TestAccuracyTop1))
        print('{}, Test Accuracy Top1 :{:.4f}, Test Loss:{:.4f}'.format(self.hardware,
                                                                        TestAccuracyTop1,
                                                                        AvgTestLoss))
        self.writerTf.add_scalar('TestAccuracyTop1', TestAccuracyTop1)
        self.writerTf.add_scalar('ValLAvgTestLoss', TestAccuracyTop1)
        self.writerTf.close()


def arrange_local_epoch_from_round(global_round=0, local_epoch_number=10):
    return global_round * local_epoch_number, (global_round + 1) * local_epoch_number
