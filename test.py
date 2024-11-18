import os
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from config.config import get_cfg
from dataset import build_data_loader

from CoSODNet import CoSODNet
import transforms as trans


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser("CoSOD_Test", add_help=False)
    parser.add_argument("-config_file", default="./config/cosod.yaml", metavar="FILE",
                        help="path to config file")
    parser.add_argument("-num_works", default=1, type=int)
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-device_id", type=str, default="0")
    parser.add_argument("-img_size", type=int, default=256)
    parser.add_argument("-max_num", type=int, default=13)
    parser.add_argument("-model_root_dir", default="./checkpoints")
    parser.add_argument("-test_data_root", type=str, default="./dataset/test_data")
    parser.add_argument("-test_datasets", nargs='+', default=["CoCA", "CoSal2015", "CoSOD3k"])
    parser.add_argument("-test_models", nargs='+', default=["duts_coco9k.pth", "duts_cocoseg.pth"])
    parser.add_argument("-save_dir", type=str, default='./predictions')
    return parser


def _get_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    return cfg


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
    model = CoSODNet(cfg)
    model.cuda()

    test_models = args.test_models
    model_dir = args.model_root_dir
    
    test_loaders = build_data_loader(args, mode='test')

    for test_iter in test_models:
        test_model_dir = os.path.join(model_dir, test_iter)
        model.load_state_dict(torch.load(test_model_dir))
        print("Model loaded from {}".format(test_model_dir))

        for dataset, data_loader in test_loaders.items():
            save_root = os.path.join(args.save_dir, dataset, 'CONDA_{}'.format(test_iter[:-4]))
            print("testing on {}".format(dataset))

            for idx, group_data in enumerate(data_loader):
                print('{}/{}'.format(idx, len(data_loader)))

                max_num = args.max_num

                flag = True
                while flag:
                    try:
                        test_group(model, group_data, save_root, max_num)
                        flag = False
                    except:
                        print("set max_num as {}".format(max_num-2))
                        max_num = max_num - 1
                        continue


if __name__ == '__main__':
    ap = argparse.ArgumentParser("CoSOD testing script", parents=[get_args_parser()])
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    cudnn.benchmark = True
    main(args)
    pass

