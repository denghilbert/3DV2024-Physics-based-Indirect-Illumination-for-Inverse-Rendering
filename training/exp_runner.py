import sys
sys.path.append('../code')
import argparse
import GPUtil

from training.train_dip import DIPTrainRunner
from training.train_geometry import GeometryTrainRunner

import torch
import numpy as np
import random


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--exps_folder_name', type=str, default='')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--trainstage', type=str, default='IDR', help='')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when training')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--max_niter', type=int, default=200001, help='max number of iterations to train for')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--if_indirect', default=False, action="store_true",
                        help='If set, render with indirect light.')
    parser.add_argument('--if_silhouette', default=False, action="store_true",
                        help='If set, render with if_silhouette light.')
    parser.add_argument('--if_comb', default=False, action="store_true",
                        help='Add generalization for combination.')
    parser.add_argument('--if_dist_weight', default=False, action="store_true",
                        help='lgt source weight.')
    parser.add_argument('--unet', default=False, action="store_true",
                        help='use unet for albedo and roughness')
    parser.add_argument('--real_world', default=False, action="store_true",
                        help='use real world dataset?')
    parser.add_argument('--which_real', type=str, default='')

    opt = parser.parse_args()


    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )

    runder_dict = {
        'DIP': DIPTrainRunner,
        'geometry': GeometryTrainRunner,
    }

    trainrunner = runder_dict[opt.trainstage](conf=opt.conf,
                                            exps_folder_name=opt.exps_folder_name,
                                            expname=opt.expname,
                                            data_split_dir=opt.data_split_dir,
                                            frame_skip=opt.frame_skip,
                                            batch_size=opt.batch_size,
                                            max_niters=opt.max_niter,
                                            is_continue=opt.is_continue,
                                            timestamp=opt.timestamp,
                                            checkpoint=opt.checkpoint,
                                            exp_dir=opt.exp_dir,
                                            local_rank=opt.local_rank,
                                            if_indirect=opt.if_indirect,
                                            if_silhouette=opt.if_silhouette,
                                            if_comb=opt.if_comb,
                                            if_dist_weight=opt.if_dist_weight,
                                            unet=opt.unet,
                                            real_world=opt.real_world,
                                            which_real=opt.which_real,)

    trainrunner.run()