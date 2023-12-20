from __future__ import print_function

import os
import sys
from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train_edge import parse_args

from config import data_root, model_root


class Evaluator(object):
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
        val_dataset = get_segmentation_dataset(args.dataset, root=data_root, split='val', mode='ms_val',
                                               transform=input_transform)
        self.num_class = val_dataset.num_class
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            root=model_root,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
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
        synchronize()
        return mIoU


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
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

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
