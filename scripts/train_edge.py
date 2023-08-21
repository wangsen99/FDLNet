import argparse
import time
import datetime
import os
import shutil
import sys
import random
import numpy as np
from tqdm import tqdm
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.visualize import get_color_pallete
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
from datetime import datetime, timedelta
from config import data_root, model_root

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fdlnet',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='night',
                        help='dataset name')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=12,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--joint_edgeseg_loss', action='store_true', default=False,
                    help='joint loss')
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='flip for test')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Segmentation loss weight for joint loss')        
    parser.add_argument('--edge_weight', type=float, default=0.01,
                        help='Edge loss weight for joint loss')
    parser.add_argument('--l2_weight', type=float, default=0,
                        help='Edge l2loss weight for joint loss')
    parser.add_argument('--att_weight', type=float, default=0.01,
                        help='Attention loss weight for joint loss')   

    parser.add_argument('--manual_seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save_dir', default='../runs/ckpt',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_epoch', type=int, default=20,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log_dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_iter', type=int, default=20,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val_epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 260,
            'sbu': 160,
            'night':260
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
            'night': 5e-3
        }
        args.lr = lrs[args.dataset.lower()]
    return args

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.flip = args.flip
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, root=data_root, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, root=data_root, split='val', mode='ms_val', transform=input_transform)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch
        val_batch_size = 1
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, val_batch_size)
        self.num_class = val_dataset.num_class
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create criterion
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d, pretrained_base=True, criterion=self.criterion).to(self.device)

        num_params = sum([param.nelement() for param in self.model.parameters()])
        logger.info('Model params = {}M'.format(num_params / 1000000))

        params_list = []
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'state_dict' in checkpoint:
                    loaded_dict = checkpoint['state_dict']                            
                    net_state_dict = self.model.state_dict()
                    new_loaded_dict = {}
                    for k in net_state_dict:
                        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
                            new_loaded_dict[k] = loaded_dict[k]
                        else:
                            print('Skipped loading parameter {}'.format(k))
                    net_state_dict.update(new_loaded_dict)
                    self.model.load_state_dict(net_state_dict)
            else:
                print("file not exist")

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
   
    def train_edge(self):
        train_main_loss = AverageMeter()
        train_seg_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        train_att_loss = AverageMeter()
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, edge, _) in enumerate(self.train_loader):
            batch_pixel_size = images.size(0) * images.size(2) * images.size(3)

            images, targets, edge = images.cuda(), targets.cuda(), edge.cuda()
            # print(images.shape, targets.shape)
            iteration = iteration + 1
            self.optimizer.zero_grad()

            main_loss = None
            loss_dict = self.model(images, gts=(targets, edge))

            if args.seg_weight > 0:
                log_seg_loss = loss_dict['seg_loss'].mean().clone().detach_()
                train_seg_loss.update(log_seg_loss.item(), batch_pixel_size)
                main_loss = loss_dict['seg_loss']

            if args.aux_weight > 0:
                log_aux_loss = loss_dict['aux_loss'].mean().clone().detach_()
                train_aux_loss.update(log_aux_loss.item(), batch_pixel_size)
                main_loss += loss_dict['aux_loss']

            
            if args.att_weight > 0:
                log_att_loss = loss_dict['att_loss'].mean().clone().detach_()
                train_att_loss.update(log_att_loss.item(), batch_pixel_size)
                main_loss += loss_dict['att_loss']


            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

            train_main_loss.update(log_main_loss.item(), batch_pixel_size)

            main_loss.backward()

            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(timedelta(seconds=int(eta_seconds)))

            a = train_main_loss.avg
            b = train_seg_loss.avg
            e = train_att_loss.avg
            f = train_aux_loss.avg

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || main loss: {:.4f} || seg loss: {:.4f} || aux loss: {:.4f} || att loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], a, b, f, e,
                        str(timedelta(seconds=int(time.time() - start_time))), eta_string))

            if not self.args.skip_val and iteration % val_per_iters == 0:
                # mIoU = self.validation_edge(iteration)
                mIoU = self.ms_val(iteration)
                self.model.train()

            self.lr_scheduler.step()

        save_checkpoint(self.model, self.optimizer, self.args, iteration, mIoU, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation_edge(self, iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        mIoU = 0
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, edge, filename) in enumerate(tqdm(self.val_loader, ascii= True)):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                seg_out, v = model(image)
            self.metric.update(seg_out[0], target)

        pixAcc, IoU, mIoU, this = self.metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.6f}".format(i + 1, pixAcc, mIoU))
        IoU = IoU.detach().numpy()
        num = IoU.size
        di = dict(zip(range(num), IoU))
        for k, v in di.items():
            logger.info("{}: {}".format(k, v))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.optimizer, self.args, iteration, mIoU, is_best)
        synchronize()
        return mIoU

    def ms_val(self, iteration):
        is_best = False
        mIoU = 0
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, batch_data in enumerate(tqdm(self.val_loader, ascii= True)):

            img_resized_list = batch_data['img_data']
            target = batch_data['seg_label']
            filename = batch_data['info']
            size = target.size()[-2:]

            with torch.no_grad():
                segSize = (target.shape[1], target.shape[2])
                scores = torch.zeros(1, self.num_class, segSize[0], segSize[1]).to(self.device).detach()
                for image in img_resized_list:
                    image = image.to(self.device)
                    target = target.to(self.device)
                    a, b = model(image)
                    logits = a
                    logits = F.interpolate(logits, size=size,
                                           mode='bilinear', align_corners=True)
                    scores += torch.softmax(logits, dim=1)
                    # scores = scores + outimg / 6
                    if self.flip:
                        # print('use flip')
                        image = torch.flip(image, dims=(3,))
                        a, b = model(image)
                        logits = a
                        logits = torch.flip(logits, dims=(3,))
                        logits = F.interpolate(logits, size=size,
                                               mode='bilinear', align_corners=True)
                        scores += torch.softmax(logits, dim=1)

            self.metric.update(scores, target)

        pixAcc, IoU, mIoU = self.metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.6f}".format(i + 1, pixAcc, mIoU))
        IoU = IoU.detach().numpy()
        num = IoU.size
        di = dict(zip(range(num), IoU))
        for k, v in di.items():
            logger.info("{}: {}".format(k, v))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.optimizer, self.args, iteration, mIoU, is_best)
        synchronize()
        return mIoU

def save_checkpoint(model, optimizer, args, iteration, mIoU, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    # print(iteration, args.iters_per_epoch)
    # print(mIoU)
    epoch = int(iteration // args.iters_per_epoch)
    if 'mean_iu' in args.last_record:
        if not os.path.exists(directory):
            os.makedirs(directory)
        last_snapshot = 'last_{}_{}_{}_epoch_{}_mean_iu_{:.5f}.pth'.format(args.model, args.backbone, args.dataset, args.last_record['epoch'], args.last_record['mean_iu'])
        last_snapshot = os.path.join(directory, last_snapshot)
        try:
            os.remove(last_snapshot)
        except OSError:
            pass
    last_snapshot = 'last_{}_{}_{}_epoch_{}_mean_iu_{:.5f}.pth'.format(args.model, args.backbone, args.dataset, epoch, mIoU)
    last_snapshot = os.path.join(directory, last_snapshot)
    args.last_record['mean_iu'] = mIoU
    args.last_record['epoch'] = epoch

    if args.distributed:
        model = model.module
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),  
    },last_snapshot)

    if is_best:
        if args.best_record['epoch'] != -1:
            best_snapshot = 'best_{}_{}_{}_epoch_{}_mean_iu_{:.5f}.pth'.format(args.model, args.backbone, args.dataset, args.best_record['epoch'], args.best_record['mean_iu'])
            best_snapshot = os.path.join(directory, best_snapshot)
            assert os.path.exists(best_snapshot), \
                'cant find old snapshot {}'.format(best_snapshot)
            os.remove(best_snapshot)

        args.best_record['epoch'] = epoch
        args.best_record['mean_iu'] = mIoU
        best_snapshot = 'best_{}_{}_{}_epoch_{}_mean_iu_{:.5f}.pth'.format(args.model, args.backbone, args.dataset, args.best_record['epoch'], args.best_record['mean_iu'])
        best_snapshot = os.path.join(directory, best_snapshot)
        shutil.copyfile(last_snapshot, best_snapshot)


if __name__ == '__main__':
    args = parse_args()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    # reference maskrcnn-benchmark

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    # args.manual_seed = random.randint(0,99999)
    args.manual_seed = 40171
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus
    args.last_record = {}
    args.best_record = {'epoch': -1, 'mean_iu': 0}
    name = 'test'
    logger = setup_logger(name, args.log_dir, get_rank(), filename='{}_{}_{}_{}.txt'.format(
        args.model, args.backbone, args.dataset, args.date_str))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    trainer = Trainer(args)
    trainer.train_edge()

    torch.cuda.empty_cache()
