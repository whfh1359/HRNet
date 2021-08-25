from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import *
import cv2, math, imutils
import dataset
import models
import numpy as np
import matplotlib.pyplot as plt
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--img-file',
                        help='input your test img',
                        type=str,
                        default='')
    parser.add_argument('--imgdir',
                        help='input your test img folder',
                        type=str,
                        default='')

    parser.add_argument('--savedir',
                        help='input your test img folder',
                        type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def main():
    # csv_path = '/home/crescom/python_work/deep-high-resolution-net.pytorch-master/point_csv/c_spine_0.1_background_fix_total.csv'
    save_csv_flie_name = 'c_kneebone_fix_total_add_prob_addL.csv'
    class_name = ['right_medial','right_f','left_medial','left_f','letter_R','letter_L']
    # prob_set = ['l1e_prob', 'l2e_prob', 'l3e_prob', 'l4e_prob', 'l5e_prob', 'l6e_prob', 'l1c_prob', 'l2c_prob',
    #             'l3c_prob', 'l4c_prob', 'l5c_prob', 'l6c_prob']

    background_point_csv = {}

    # f = open(csv_path, 'r', encoding='utf-8')
    # rdr = csv.reader(f)
    # for line in rdr:
    #     if line[0] != 'file_name':
    #         background_point_csv[line[0]] = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
    # f.close()

    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    #### get weight
    # get_weight = []
    # for param in model.parameters():
    #     get_weight.append(param.data)

    # print('len layer', model.layer[0])
    # for i in range(len(model.layer)):
    #     weight = model.layer[0].weight
    #     print(weight)

    # sd = model.state_dict()
    # print(sd.keys())

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    model.eval()

    # Loading an image
    # image_file = args.img_file
    img_dir = args.imgdir
    img_list = [i.replace('.png','').replace('.jpg','') for i in os.listdir(img_dir)]

    idx_set = []
    for i in class_name:
        idx_set.append(i + '_x')
        idx_set.append(i + '_y')
    idx_set.insert(0,'file_name')

    # [idx_set.append(i) for i in prob_set]
    print(idx_set)

    f = open(args.savedir + '/' + save_csv_flie_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(idx_set)

    with torch.no_grad():
        for one_img in img_list: # ['01233898_2014-08-05_RSC05302_L_3']:
            image_file = img_dir + '/' + one_img + '.jpg'
            print(image_file)

            data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if data_numpy is None:
                logger.error('=> fail to read {}'.format(image_file))
                raise ValueError('=> fail to read {}'.format(image_file))

            ##### object detection box crop
            # print(background_point_csv[one_img])
            #xmin, ymin, xmax, ymax = background_point_csv[one_img][0], background_point_csv[one_img][1], background_point_csv[one_img][2], background_point_csv[one_img][3]
            #box = [xmin, ymin, xmax - xmin, ymax - ymin]
            box = [0,0,data_numpy.shape[1],data_numpy.shape[0]] #확인필
            ##### full image crop
            # box = [0, 0, data_numpy.shape[1], data_numpy.shape[0]]
            print(box)
            # box = [123, 197, 963-123, 1597-197]
            c, s = _box2cs(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            r = 0

            trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
            input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            input = transform(input).unsqueeze(0)
            # switch to evaluate mode


            # compute output heatmap
            output = model(input)
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
            print(maxvals)

            point_save_list = []

            for one_disc_point_num in range(len(preds[0])):
                x, y = int(preds[0][one_disc_point_num][0]), int(preds[0][one_disc_point_num][1])
                point_save_list.append(x)
                point_save_list.append(y)
            for one_disc_point_num in range(maxvals.shape[1]):
                prob = maxvals[0][one_disc_point_num][0]
                print(prob)
                point_save_list.append(prob)

            point_save_list.insert(0, one_img.replace('.jpg',''))

            wr.writerow(point_save_list)
            print('point_save_list', point_save_list)

    f.close()

if __name__ == '__main__':
    main()