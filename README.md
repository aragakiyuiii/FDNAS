# Federated Direct Neural Architecture Search

This repository accompanies the paper [Towards Tailored Models on Private AIoT Devices: Federated Direct Neural Architecture Search](https://www.researchgate.net/publication/358814243_Towards_Tailored_Models_on_Private_AIoT_Devices_Federated_Direct_Neural_Architecture_Search) and contains the basic code for replicating the training and computations in it. We will continue to add more details and code to this repository as we go along

## Search
### FDNAS:
```sh
nohup python federated_main.py  --gpu 2 --search --warmup --resume --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 128 --test_batch_size 128 --init_lr 0.025 --arch_algo grad --arch_lr 0.025 > 1.out 2>&1 &
```

### Cluster FDNAS:
```sh

nohup python federated_main.py  --gpu 0 --search --resume --last_round 175 --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 128 --test_batch_size 128 --init_lr 0.025 --arch_algo grad --arch_lr 0.025 --target_hardware cpu --object_to_search cpu --grad_binary_mode two --grad_reg_loss_type add#linear --grad_reg_loss_lambda 0.05 > cls_cpu.out 2>&1 &


nohup python federated_main.py  --gpu 2 --search --resume --last_round 175 --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 128 --test_batch_size 128 --init_lr 0.025 --arch_algo grad --arch_lr 0.025 --target_hardware gpu8 --object_to_search gpu8 --grad_binary_mode two --grad_reg_loss_type add#linear --grad_reg_loss_lambda 0.05 > cls_gpu.out 2>&1 &
```
## Derive normal model
```sh

nohup python convert_normal_architecture.py --gpu 0 --path output/exp_name/ --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ > cvt.out 2>&1 &
```
## Train a derived model
```sh
nohup python retrain.py --path output/fednas-grad12,3,4,3,4,3/learned_net --gpu 3 --resume --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 256 --test_batch_size 256 --init_lr 0.05 > 2.out 2>&1 &

nohup python retrain.py --resume --path output/fednas-grad1cpu2,3,4,3,4,3/learned_net --object_to_retrain cpu --gpu 0  --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 128 --test_batch_size 128 --init_lr 0.025 --last_round 300 > 160to250round_cpu_retrain.out 2>&1 &

nohup python retrain.py --resume --path output/fednas-grad1gpu2,3,4,3,4,3/learned_net --object_to_retrain gpu8 --gpu 0  --dataset_location /home/grouplcheng/data/zch/dataset/cifar100/ --train_batch_size 128 --test_batch_size 128 --init_lr 0.025 --last_round 300 > 160to250round_gpu_retrain.out 2>&1 &
```