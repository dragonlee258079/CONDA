import os
import argparse
import logging
import pprint
import shutil
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import List
from bisect import bisect_right
from torch.autograd import Variable

from torch import Tensor

from config.config import get_cfg
from dataset import build_data_loader

from CoSODNet import CoSODNet

import transforms as trans

import evaluation.metric as M


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser("CoSOD_Train", add_help=False)
    parser.add_argument("-config_file", default="./config/cosod.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("-num_works", default=1, type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-device_id", type=str, default="0", help="choose cuda visiable devices")
    parser.add_argument("-train_data_root", type=str, default="./dataset/train_data")
    parser.add_argument("-use_dust_syn", type=bool, default=True)
    parser.add_argument("-use_coco9k_syn", type=bool, default=False)
    parser.add_argument("-train_datasets", type=str, default="DUTS_class+CoCo9k",
                        help="[DUTS_class+CoCo9k, DUTS_class+CoCo_Seg, DUTS_class, CoCo9k]")
    parser.add_argument("-test_datasets", nargs='+', default=["CoCA"], help="CoCA, CoSal2015, CoSOD3k")
    parser.add_argument('-test_data_root', type=str, default='./dataset/test_data')
    parser.add_argument("-save_dir", type=str, default='./predictions', help="dir for saving prediction maps")
    parser.add_argument("-model_root_dir", default="./models", help="dir for saving checkpoint")
    parser.add_argument("-max_num", type=int, default=6)
    parser.add_argument("-test_max_num", type=int, default=13)
    parser.add_argument("-img_size", type=int, default=256)
    parser.add_argument("-scale_size", type=int, default=288)
    parser.add_argument("-train_steps", type=int, default=80000)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-STEPS", nargs='+', default=[60000, 80000])
    parser.add_argument("-GAMMA", type=float, default=0.1)
    parser.add_argument("-warmup_factor", type=float, default=1.0/1000)
    parser.add_argument("-warmup_iters", type=int, default=1000)
    parser.add_argument("-warmup_method", type=str, default="linear")
    parser.add_argument("-max_epoches", type=int, default=300)
    return parser


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    fh.write('until_{}_run_iter_num{}\n'.format(epoch, whole_iter_num))
    fh.write('{}_epoch_total_loss:{}\n'.format(epoch, epoch_total_loss))
    fh.write('{}_epoch_loss:{}\n'.format(epoch, epoch_loss))
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: {}'.format(param_group['lr']))
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: {}'.format(param_group['lr']))
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('decode:update:lr{}\n'.format(update_lr_group['lr']))
    fh.write('\n')
    fh.close()


def create_logger(model_name):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if not os.path.exists('./{}/log'.format(model_name)):
        os.makedirs('./{}/log'.format(model_name), exist_ok=True)
    log_file = './{}/log/{}_.log'.format(model_name, time_str)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename=str(log_file),
        format=head,
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def _get_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg


def _get_project_save_dir(model_root_dir, model_name):
    proj_save_dir = os.path.join(model_root_dir, model_name)

    if not os.path.exists(proj_save_dir):
        os.makedirs(proj_save_dir, exist_ok=True)

    return proj_save_dir


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:

    base_params = [params for name, params in model.named_parameters()
                   if 'encoder' in name and params.requires_grad]
    other_params = [params for name, params in model.named_parameters()
                    if 'encoder' not in name]

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': args.lr * 0.01}, {'params': other_params}],
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    return optimizer


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        return self.get_lr()


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
         method (str): warmup method; either "constant" or "linear".
         iter (int): iteration at which to calculate the warmup factor.
         warmup_iters (int): the number of warmup iterations.
         warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used)

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_lr_scheduler(
        args, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    return WarmupMultiStepLR(
        optimizer,
        args.STEPS,
        args.GAMMA,
        warmup_factor=args.warmup_factor,
        warmup_iters=args.warmup_iters,
        warmup_method=args.warmup_method
    )


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.f_co_bce = 0
        self.f_iou = 0
        self.s_3_co_bce = 0
        self.s_2_co_bce = 0
        self.cyc_loss_3 = 0
        self.cyc_loss_2 = 0

    def reset_loss(self):
        self.f_co_bce = 0
        self.f_iou = 0
        self.s_3_co_bce = 0
        self.s_2_co_bce = 0
        self.cyc_loss_3 = 0
        self.cyc_loss_2 = 0

    def iou(self, pred, gt):
        pred = F.sigmoid(pred)
        N, C, H, W = pred.shape
        min_tensor = torch.where(pred < gt, pred, gt)
        max_tensor = torch.where(pred > gt, pred, gt)
        min_sum = min_tensor.view(N, C, H * W).sum(dim=2)
        max_sum = max_tensor.view(N, C, H * W).sum(dim=2)
        loss = 1 - (min_sum / max_sum).mean()
        return loss

    def stage_loss(self, s_3_co_pred, s_2_co_pred, co_gt):
        pred_3_size = s_3_co_pred.shape[2:]
        pred_2_size = s_2_co_pred.shape[2:]
        co_gt_3 = F.interpolate(co_gt, size=pred_3_size, mode="nearest")
        co_gt_2 = F.interpolate(co_gt, size=pred_2_size, mode="nearest")

        self.s_3_co_bce += self.bce(s_3_co_pred, co_gt_3)
        self.s_2_co_bce += self.bce(s_2_co_pred, co_gt_2)

    def average_loss(self, stage_num):
        self.s_3_co_bce = self.s_3_co_bce / stage_num
        self.s_2_co_bce = self.s_2_co_bce / stage_num

    def __call__(self, result, co_gt: Tensor):
        self.reset_loss()

        co_gt[co_gt < 0.5] = 0.
        co_gt[co_gt >= 0.5] = 1.

        final_pred = result.pop('final_pred')
        sals_3 = result.pop('sals_3')
        sals_2 = result.pop('sals_2')

        self.cyc_loss_3 = result.pop('cyc_loss_3')
        self.cyc_loss_2 = result.pop('cyc_loss_2')

        self.f_co_bce = self.bce(final_pred, co_gt)
        self.f_iou = self.iou(final_pred, co_gt)

        for i in range(len(sals_3)):
            self.stage_loss(
                sals_3[i], sals_2[i], co_gt
            )

        self.average_loss(len(sals_3))

        loss = self.f_co_bce + self.f_iou + self.s_3_co_bce + self.s_2_co_bce + self.cyc_loss_3 + self.cyc_loss_2

        return loss


def test_group(model, group_data, save_root, max_num):
    img_num = group_data['imgs'].shape[1]
    groups = list(range(0, img_num + 1, max_num))
    if groups[-1] != img_num:
        groups.append(img_num)

    print(groups)

    for i in range(len(groups) - 1):
        if i == len(groups) - 2:
            end = groups[i + 1]
            start = max(0, end - max_num)
        else:
            start = groups[i]
            end = groups[i + 1]

        print(start, end)

        inputs = Variable(group_data['imgs'][:, start:end].squeeze(0).cuda())
        subpaths = group_data['subpaths'][start:end]
        ori_sizes = group_data['ori_sizes'][start:end]

        with torch.no_grad():

            result = model(inputs)

            pred_prob = torch.sigmoid(result['final_pred'])

            save_final_path = os.path.join(save_root, subpaths[0][0].split('/')[0])
            os.makedirs(save_final_path, exist_ok=True)

            for p_id in range(end - start):
                pre = pred_prob[p_id, :, :, :].data.cpu()

                subpath = subpaths[p_id][0]
                ori_size = (ori_sizes[p_id][1].item(),
                            ori_sizes[p_id][0].item())

                transform = trans.Compose([
                    trans.ToPILImage(),
                    trans.Scale(ori_size)
                ])
                outputImage = transform(pre)
                filename = subpath.split('/')[1]
                outputImage.save(os.path.join(save_final_path, filename))


def main(args):
    cfg = _get_cfg(args.config_file)
    model_name = args.model_name
    if model_name is None:
        model_name = os.path.abspath('').split('/')[-1]
    proj_save_dir = _get_project_save_dir(args.model_root_dir, model_name)

    logger = create_logger(model_name)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    train_loader = build_data_loader(args, mode='train')

    logger.info('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    logger.info("=> building model")
    model = CoSODNet(cfg)

    model.cuda()
    model.train()

    logger.info(model)

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    max_epoches = args.max_epoches
    train_steps = args.train_steps

    cri = Criterion()

    whole_iter_num = 0
    for epoch in range(max_epoches):

        logger.info("Starting epoch {}/{}.".format(epoch + 1, max_epoches))
        logger.info("epoch: {} ------ lr:{}".format(epoch, optimizer.param_groups[0]['lr']))

        for iteration, data_batch in enumerate(train_loader):
            imgs = Variable(data_batch["imgs"].squeeze(0).cuda())
            co_gts = Variable(data_batch["co_gts"].squeeze(0).cuda())

            result = model(imgs, co_gts)

            loss = cri(result, co_gts)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            whole_iter_num += 1

            if whole_iter_num == train_steps:
                torch.save(
                    model.state_dict(),
                    os.path.join(proj_save_dir, 'iterations{}.pth'.format(train_steps))
                )
                break

            logger.info('Whole iter step:{0} - epoch progress:{1}/{2} - total_loss:{3:.4f} - f_co_bce:{4:.4f} '
                        '- f_iou: {5:.4f} - s_3_co_bce: {6:4f} - s_2_co_bce: {7:4f} - cyc_loss_3: {8:4f} '
                        '- cyc_loss_2: {9:4f}'
                        '- batch_size: {10}'.format(whole_iter_num, epoch, max_epoches, loss.item(), cri.f_co_bce,
                                                   cri.f_iou, cri.s_3_co_bce, cri.s_2_co_bce, cri.cyc_loss_3,
                                                   cri.cyc_loss_2, co_gts.shape[0]))

        Sm_fun = M.Smeasure()
        Em_fun = M.Emeasure()
        FM_fun = M.Fmeasure_and_FNR()
        MAE_fun = M.MAE()

        test_loaders = build_data_loader(args, mode='test')
        data_loader = test_loaders['CoCA']

        save_root = os.path.join(args.save_dir, 'CoCA', '{}_iter{}'.format(model_name, whole_iter_num))
        print("evaluating on {}".format('CoCA'))
        for idx, group_data in enumerate(data_loader):
            print('{}/{}'.format(idx, len(data_loader)))

            max_num = args.test_max_num
            flag = True
            while flag:
                try:
                    test_group(model, group_data, save_root, max_num)
                    flag = False
                except:
                    print("set max_num as {}".format(max_num-2))
                    max_num = max_num - 1
                    continue

        # pred_data_dir = os.path.join(save_root, dataset)
        label_data_dir = os.path.join(args.data_root, 'CoCA', 'GroundTruth')
        classes = os.listdir(label_data_dir)
        for k in range(len(classes)):
            print('\r{}/{}'.format(k, len(classes)), end="", flush=True)
            class_name = classes[k]
            img_list = os.listdir(os.path.join(label_data_dir, class_name))
            for l in range(len(img_list)):
                img_name = img_list[l]
                pred = cv2.imread(os.path.join(save_root, class_name, img_name), 0)
                gt = cv2.imread(os.path.join(label_data_dir, class_name, img_name[:-4]+'.png'), 0)
                Sm_fun.step(pred=pred/255, gt=gt/255)
                FM_fun.step(pred=pred/255, gt=gt/255)
                Em_fun.step(pred=pred/255, gt=gt/255)
                MAE_fun.step(pred=pred/255, gt=gt/255)

        sm = Sm_fun.get_results()['sm']
        fm = FM_fun.get_results()[0]['fm']['curve'].max()
        em = Em_fun.get_results()['em']['curve'].max()
        mae = MAE_fun.get_results()['mae']

        logger.info('\nEvaluating epoch {0} get SM {1:.4f} Fm {2:.4f} Em {3:.4f} MAE {4:.4f}'
                    .format(epoch, sm, fm, em, mae))

        if sm > 0.74:
            torch.save(
                model.state_dict(),
                os.path.join(proj_save_dir, 'iterations{}.pth'.format(whole_iter_num))
            )
        else:
            shutil.rmtree(save_root)

    torch.save(
        model.state_dict(),
        os.path.join(proj_save_dir, 'iterations{}.pth'.format(whole_iter_num))
    )

    logger.info('Epoch finished !!!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser("CoSOD training script", parents=[get_args_parser()])
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    cudnn.benchmark = True
    main(args)
