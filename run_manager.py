# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import torch.nn.parallel
import torch.optim

import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils import *
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge


# run_manager is used by nas_manager

class RunConfig(object):

    def __init__(self, client_id, dataset_location, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency, print_frequency, search, cifar10,
                 ):
        self.client_id = client_id
        self.dataset_location = dataset_location
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency
        self.search = search
        self.cifar10 = cifar10

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
            if epoch < 25:
                lr = ((epoch + 1) / 25) * self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.search:
                if self.cifar10:
                    from data_providers.cifar import CifarDataProvider
                    self._data_provider = CifarDataProvider(**self.data_config)
                else:
                    # use cifar100
                    from data_providers.cifar import CifarDataProvider100
                    self._data_provider = CifarDataProvider100(**self.data_config)
            else:
                if self.cifar10:
                    from data_providers.cifar_test import CifarDataProvider
                    self._data_provider = CifarDataProvider(**self.data_config)
                else:
                    # use cifar100
                    from data_providers.cifar_test import CifarDataProvider100
                    self._data_provider = CifarDataProvider100(**self.data_config)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train()

    @property
    def valid_loader(self):
        return self.data_provider.valid()

    @property
    def test_loader(self):
        return self.data_provider.test()

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0},
                ], lr=self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov,
                                            weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True, measure_latency=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0

        self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency model
        self.net_on_cpu_for_latency = copy.deepcopy(self.net).cpu()
        self.latency_estimator = LatencyEstimator()
        self.start_epoch = 0
        self.round = 0
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.net = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            print("can not use GPU!")
            raise ValueError
        # net info
        self.print_net_info(measure_latency)

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                self.net.module.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.module.get_parameters(keys, mode='include'),  # parameters without weight decay
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters())

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop

    def cmp_lat(self, batch_size=64, iterations=10):
        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)
        with torch.no_grad():
            images = torch.zeros(data_shape)
            # GPU-latency
            images.cuda()
            print('=========GPU Speed Testing=========')
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                output = self.net(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            gpu_latency = elapsed_time_ms / iterations
            torch.cuda.empty_cache()
            print('=========CPU Speed Testing=========')
            images.cpu()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                output = self.net_on_cpu_for_latency(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            cpu_latency = elapsed_time_ms / iterations
            torch.cuda.empty_cache()
            print('GPU latency:', gpu_latency, 'CPU latency:', cpu_latency)

    # noinspection PyUnresolvedReferences
    def net_latency(self, l_type='gpu4', fast=True, given_net=None):
        if 'gpu' in l_type:
            l_type, batch_size = l_type[:3], int(l_type[3:])
        else:
            batch_size = 1

        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)

        if given_net is not None:
            net = given_net
        else:
            net = self.net.module  # 并行后的修改
            # net = self.net

        if l_type == 'mobile':
            predicted_latency = 0
            try:
                assert isinstance(net, ProxylessNASNets)
                # first conv
                predicted_latency += self.latency_estimator.predict(
                    'Conv', [224, 224, 3], [112, 112, net.first_conv.out_channels]
                )
                # feature mix layer
                predicted_latency += self.latency_estimator.predict(
                    'Conv_1', [7, 7, net.feature_mix_layer.in_channels], [7, 7, net.feature_mix_layer.out_channels]
                )
                # classifier
                predicted_latency += self.latency_estimator.predict(
                    'Logits', [7, 7, net.classifier.in_features], [net.classifier.out_features]  # 1000
                )
                # blocks
                fsize = 112
                for block in net.blocks:
                    mb_conv = block.mobile_inverted_conv
                    shortcut = block.shortcut
                    if isinstance(mb_conv, MixedEdge):
                        mb_conv = mb_conv.active_op
                    if isinstance(shortcut, MixedEdge):
                        shortcut = shortcut.active_op

                    if mb_conv.is_zero_layer():
                        continue
                    if shortcut is None or shortcut.is_zero_layer():
                        idskip = 0
                    else:
                        idskip = 1
                    out_fz = fsize // mb_conv.stride
                    block_latency = self.latency_estimator.predict(
                        'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    predicted_latency += block_latency
                    fsize = out_fz
            except Exception:
                predicted_latency = 200
                print('fail to predict the mobile latency')
            return predicted_latency, None
        elif l_type == 'cpu':
            if fast:
                n_warmup = 1
                n_sample = 2
            else:
                n_warmup = 10
                n_sample = 100
            try:
                self.net_on_cpu_for_latency.set_active_via_net(net)
            except AttributeError:
                print(type(self.net_on_cpu_for_latency), ' do not `support set_active_via_net()`')
            net = self.net_on_cpu_for_latency
            images = torch.zeros(data_shape, device=torch.device('cpu'))
        elif l_type == 'gpu':
            if fast:
                n_warmup = 5
                n_sample = 10
            else:
                n_warmup = 50
                n_sample = 100
            images = torch.zeros(data_shape, device=self.device)
        else:
            # raise NotImplementedError
            return 0, 0  # latency = 0 when Hardware==None

        measured_latency = {'warmup': [], 'sample': []}
        net.eval()
        with torch.no_grad():
            for i in range(n_warmup + n_sample):
                start_time = time.time()
                net(images)
                used_time = (time.time() - start_time) * 1e3  # ms
                if i >= n_warmup:
                    measured_latency['sample'].append(used_time)
                else:
                    measured_latency['warmup'].append(used_time)
        net.train()
        return sum(measured_latency['sample']) / n_sample, measured_latency

    def print_net_info(self, measure_latency=None):
        # print(self.net)
        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }

        # flops
        flops = self.net_flops()
        if self.out_log:
            print('Total FLOPs: %.1fM' % (flops / 1e6))
        net_info['flops'] = '%.1fM' % (flops / 1e6)

        # latency
        latency_types = [] if measure_latency is None else measure_latency.split('#')
        for l_type in latency_types:
            latency, measured_latency = self.net_latency(l_type, fast=False, given_net=None)
            if self.out_log:
                print('Estimated %s latency: %.3fms' % (l_type, latency))
            net_info['%s latency' % l_type] = {
                'val': latency,
                'hist': measured_latency
            }
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None, HardwareClass=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_clients_opt(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        print('the latest_fname is: ', latest_fname)
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        print("=> loading checkpoint '{}'".format(model_fname))

        checkpoint = torch.load(model_fname, map_location=torch.device('cpu'))  # reduce max gpu-cost

        if str(self.run_config.data_provider.client_id) + '_weight_optimizer' in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint[str(self.run_config.data_provider.client_id) + '_weight_optimizer'])
        print("=> loaded opt checkpoint '{}'".format(model_fname))

    def return_model_dict(self, ):
        _w = self.net.module.state_dict()
        return _w

    def get_run_manager_model(self):
        return self.net.module

    def get_local_data_weight(self):
        return self.run_config.data_provider.trn_set_length

    def load_model(self, model_fname=None, HardwareClass=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/' % self.save_path + 'checkpoint.pth.tar'
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            checkpoint = torch.load(model_fname, map_location='cpu')
            # self.net.load_state_dict(checkpoint['state_dict'])  # 并行后的修改
            self.net.module.load_state_dict(checkpoint['state_dict'])  # 并行后的修改
            if 'round' in checkpoint:
                self.round = checkpoint['round'] + 1
            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception as e:
            print('Exception about load model:', e)
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4, default=set_to_list)  # 并行后的修改

        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4, default=set_to_list)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
            print("USE: test_loader")
        else:
            data_loader = self.run_config.valid_loader
        data_loader = list(data_loader)
        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()

        net = net.cuda()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                # 9.to(device,non_blocking=True)
                # compute output
                output = net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train_run_manager_one_epoch(self, adjust_lr_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        lr = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            lr.update(new_lr)

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        return top1, top5, losses, lr.avg

    def train_run_manager(self, start_local_epoch=None, last_local_epoch=None, print_top5=False, server_model=None,
                          writer=None):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.module.load_state_dict(server_model.module.state_dict())
            else:
                self.net.module.load_state_dict(server_model.state_dict())

        nBatch = len(self.run_config.train_loader)

        start_epoch = start_local_epoch
        n_epochs = last_local_epoch
        val_loss, val_acc, val_acc5, lr = None, None, None, None
        train_acc_top1_arr, train_acc_top5_arr, train_losses_arr = AverageMeter(), AverageMeter(), AverageMeter()
        for epoch in range(start_epoch, n_epochs):
            train_acc_top1, train_acc_top5, train_losses, lr = self.train_run_manager_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch)
            )
            train_acc_top1_arr.update(train_acc_top1.avg)
            train_acc_top5_arr.update(train_acc_top5.avg)
            train_losses_arr.update(train_losses.avg)
            writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_trn_top1',
                              train_acc_top1.avg,
                              epoch)
            writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_trn_top5',
                              train_acc_top5.avg,
                              epoch)
            writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_trn_loss',
                              train_losses.avg,
                              epoch)

            if epoch % 3 == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False, return_top5=True)
                writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_val_loss',
                                  val_loss,
                                  epoch)
                writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_val_top1',
                                  val_acc,
                                  epoch)
                writer.add_scalar('client_' + str(self.run_config.data_provider.client_id) + '_val_top5',
                                  val_acc5,
                                  epoch)
        return train_losses_arr.avg, train_acc_top1_arr.avg, train_acc_top5_arr.avg, val_loss, val_acc, val_acc5, lr

    def inference_run_manager(self, server_model=None, is_test=True):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.module.load_state_dict(server_model.module.state_dict())
            else:
                self.net.module.load_state_dict(server_model.state_dict())
        else:
            print("WARNING: Have no server model")
        val_loss, val_top1, val_top5 = self.validate(is_test=True, return_top5=True)
        return val_loss, val_top1, val_top5


class CifarRunConfig(RunConfig):

    def __init__(self, client_id=10, dataset_location=None, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine',
                 lr_schedule_param=None,
                 dataset='cifar', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=8, search=True, cifar10=True,
                 **kwargs):
        super(CifarRunConfig, self).__init__(
            client_id, dataset_location, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency, search, cifar10,
        )
        self.dataset_location = dataset_location
        self.n_worker = n_worker

    @property
    def data_config(self):
        return {
            'client_id': self.client_id,
            'dataset_location': self.dataset_location,
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
