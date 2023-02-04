import sys

sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
import math
import time
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from model.sg_render import compute_envmap
import imageio

tonemap_img = lambda x: np.power(x, 1. / 2.2)
clip_img = lambda x: np.clip(x, 0., 1.)


def decode_img(img, batch_size, total_pixels, img_res, is_tonemap=False):
    img = img.reshape(batch_size, total_pixels, 3)
    img = plt.lin2img(img, img_res).detach().cpu().numpy()[0]
    img = img.transpose(1, 2, 0)
    if is_tonemap:
        img = tonemap_img(img)
    img = clip_img(img)
    return img

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

import lpips
import skimage
loss_fn_vgg = lpips.LPIPS(net='vgg')
def relit_with_light(model, relit_dataloader, images_dir, total_pixels, img_res,
                     albedo_ratio=None, light_type='origin', trainstage='geometry'):
    all_frames = []
    psnrs = []
    ssims = []
    lpipss = []
    time_elapses = []
    f = open(os.path.join(images_dir, 'eval.txt'), 'w')
    for data_index, (indices, model_input, ground_truth) in enumerate(relit_dataloader):
        if data_index != 0: continue
        print('relighting data_index: ', data_index, len(relit_dataloader))
        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []


        count = 0
        start = time.time()
        for s0 in split:
            s0['albedo_ratio'] = albedo_ratio
            ## for memory debugging
            # print(count, (s0['object_mask']==True).sum())
            # print(torch.cuda.memory_allocated() / (1024**2))
            # count += 1
            out = model(s0, trainstage=trainstage, if_testing=True)
            if trainstage == 'geometry':
                res.append({
                    'normals': out['normals'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach(),
                    'idr_rgb': out['idr_rgb'].detach(),
                })
            else:
                res.append({
                    'normals': out['normals'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach(),
                    'roughness': out['roughness'].detach(),
                    'diffuse_albedo': out['diffuse_albedo'].detach(),
                    'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
                    'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                    'indir_rgb': out['indir_rgb'].detach(),
                    'sg_rgb': out['sg_rgb'].detach(),
                    'bg_rgb': out['bg_rgb'].detach(),
                })
        end = time.time()
        time_elapse = end - start


        out_img_name = '{}'.format(indices[0])
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        assert (batch_size == 1)
        if trainstage == 'geometry':
            object_mask = model_outputs['network_object_mask'].unsqueeze(0)
            object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
            rgb_relit = model_outputs['idr_rgb']
            rgb_relit[rgb_relit<0] = 0
            rgb_relit = decode_img(rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)
            bg_mask = ~object_mask.expand(-1, -1, 3).cpu().numpy()
            rgb_relit_env_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
            rgb_relit_env_bg.save('{0}/sg_rgb_bg_{1}.png'.format(images_dir, out_img_name))

            normal = model_outputs['normals']
            normal = (normal + 1.) / 2.
            normal = decode_img(normal, batch_size, total_pixels, img_res, is_tonemap=False)
            normal = Image.fromarray((normal * 255).astype(np.uint8))
            normal.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))

            rgb_gt = decode_img(ground_truth['rgb'], batch_size, total_pixels, img_res, is_tonemap=True)
            object_mask = decode_img(object_mask.reshape(-1, 1).unsqueeze(0).expand(ground_truth['rgb'].shape),
                                     batch_size, total_pixels, img_res, is_tonemap=True)
            rgb_eval_masked = rgb_relit * object_mask
            rgb_gt_masked = rgb_gt * object_mask
            psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, object_mask)
            psnrs.append(psnr)
            ssim = skimage.measure.compare_ssim(rgb_eval_masked, rgb_gt_masked, data_range=1, multichannel=True)
            ssims.append(ssim)
            lpips = loss_fn_vgg(
                torch.tensor(rgb_eval_masked.reshape(rgb_eval_masked.shape[2], rgb_eval_masked.shape[0],
                                                     rgb_eval_masked.shape[1])).to(torch.float32),
                torch.tensor(rgb_gt_masked.reshape(rgb_eval_masked.shape[2], rgb_eval_masked.shape[0],
                                                   rgb_eval_masked.shape[1])).to(torch.float32)
            )
            lpipss.append(lpips)
            time_elapses.append(time_elapse)
            print('psnr: {0}, ssim: {1}, lpips: {2}, time_elapse {3}'.format(psnr, ssim, lpips, time_elapse))
            f.write(str(data_index) + '\n')
            f.write('psnr: {0}, ssim: {1}, lpips: {2}'.format(psnr, ssim, lpips) + '\n')
            continue
        # input mask
        mask = model_input['object_mask']
        mask = plt.lin2img(mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)

        # render background
        bg_rgb = model_outputs['bg_rgb']
        bg_rgb = decode_img(bg_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
        object_mask = model_outputs['network_object_mask'].unsqueeze(0)
        object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
        ### save sg

        if light_type == 'origin':
            rgb_relit = model_outputs['sg_rgb'] + model_outputs['indir_rgb']
        else:
            rgb_relit = model_outputs['sg_rgb']
        rgb_relit = decode_img(rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)
        rgb_gt = decode_img(ground_truth['rgb'], batch_size, total_pixels, img_res, is_tonemap=True)

        # 为了air
        object_mask = object_mask * torch.tensor(rgb_gt[:, :, :1]).cuda().bool()

        # envmap background
        bg_mask = ~object_mask.expand(-1, -1, 3).cpu().numpy()
        rgb_relit[bg_mask] = bg_rgb[bg_mask]
        rgb_relit_env_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
        rgb_relit_env_bg.save('{0}/sg_rgb_bg_{1}.png'.format(images_dir, out_img_name))
        all_frames.append(np.array(rgb_relit))

        # validation with some metrics
        object_mask = decode_img(object_mask.reshape(-1,1).unsqueeze(0).expand(ground_truth['rgb'].shape), batch_size, total_pixels, img_res, is_tonemap=True)
        rgb_eval_masked = rgb_relit * object_mask
        rgb_gt_masked = rgb_gt * object_mask
        psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, object_mask)
        psnrs.append(psnr)
        ssim = skimage.measure.compare_ssim(rgb_eval_masked, rgb_gt_masked, data_range=1, multichannel=True)
        ssims.append(ssim)
        lpips = loss_fn_vgg(
            torch.tensor(rgb_eval_masked.reshape(rgb_eval_masked.shape[2], rgb_eval_masked.shape[0], rgb_eval_masked.shape[1])).to(torch.float32),
            torch.tensor(rgb_gt_masked.reshape(rgb_eval_masked.shape[2], rgb_eval_masked.shape[0], rgb_eval_masked.shape[1])).to(torch.float32)
        )
        lpipss.append(lpips[0][0])
        time_elapses.append(time_elapse)

        print('psnr: {0}, ssim: {1}, lpips: {2}, time_elapse {3}'.format(psnr, ssim, lpips, time_elapse))


        if light_type == 'origin':
            ### save roughness
            roughness_relit = model_outputs['roughness']
            roughness_relit = decode_img(roughness_relit, batch_size, total_pixels, img_res, is_tonemap=False)
            roughness_relit = Image.fromarray((roughness_relit * 255).astype(np.uint8))
            roughness_relit.save('{0}/roughness_{1}.png'.format(images_dir, out_img_name))

            ### save diffuse albedo
            albedo_relit = model_outputs['diffuse_albedo']
            albedo_relit = decode_img(albedo_relit, batch_size, total_pixels, img_res, is_tonemap=True)
            albedo_relit = Image.fromarray((albedo_relit * 255).astype(np.uint8))
            albedo_relit.save('{0}/albedo_{1}.png'.format(images_dir, out_img_name))

            ### save normals
            normal = model_outputs['normals']
            normal = (normal + 1.) / 2.
            normal = decode_img(normal, batch_size, total_pixels, img_res, is_tonemap=False)
            normal = Image.fromarray((normal * 255).astype(np.uint8))
            normal.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))

            ### save indirect rendering
            # indir_rgb = model_outputs['indir_rgb']
            # indir_rgb = decode_img(indir_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
            # indir_rgb = Image.fromarray((indir_rgb * 255).astype(np.uint8))
            # indir_rgb.save('{0}/sg_indir_rgb_{1}.png'.format(images_dir, out_img_name))

        f.write(str(data_index) + '\n')
        f.write('psnr: {0}, ssim: {1}, lpips: {2}, time_elapse {3}'.format(psnr, ssim, lpips, time_elapse) + '\n')

    f.write('avg_psnr: {0}, avg_ssim: {1}, avg_lpips: {2}, avg_time_elapse {3}'.format(
        sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(lpipss) / len(lpipss),
        sum(time_elapses) / len(time_elapses)) + '\n')
    print(sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(lpipss) / len(lpipss), sum(time_elapses) / len(time_elapses))
    f.close()
    imageio.mimwrite(os.path.join(images_dir, 'video_rgb.mp4'), all_frames, fps=20, quality=9)

    print('Done rendering', images_dir)


def relight_obj(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    relits_folder_name = kwargs['relits_folder_name']
    exp_dir = kwargs['exp_dir']
    local_rank = kwargs['local_rank']
    trainstage = kwargs['trainstage']
    if_dist_weight = kwargs['if_dist_weight']
    unet = kwargs['unet']
    real_world = kwargs['real_world']
    which_real = kwargs['which_real']

    if trainstage == 'IDR':
        expname = 'after_geometry-' + kwargs['expname']
    elif trainstage == 'geometry':
        expname = 'geometry-' + kwargs['expname']
    if_indirect = kwargs['if_indirect']
    if_silhouette = kwargs['if_silhouette']


    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join(exp_dir, kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join(exp_dir, kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join(exp_dir, relits_folder_name))
    expdir = os.path.join(exp_dir, exps_folder_name, expname)
    relitdir = os.path.join(exp_dir, relits_folder_name, expname, os.path.basename(kwargs['data_split_dir']))

    model = utils.get_class(conf.get_string('train.model_class'))\
        (conf=conf.get_config('model'), if_indirect=if_indirect, if_silhouette=if_silhouette, unet=unet)
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True)

    # load data
    if real_world:
        if which_real == 'DTU':
            relit_dataset = utils.get_class(conf.get_string('train.dataset_real_world')) \
                (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        elif which_real == 'colmap':
            relit_dataset = utils.get_class(conf.get_string('train.dataset_colmap')) \
                (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
    else:
        relit_dataset = utils.get_class(conf.get_string('train.dataset_class'))(
            kwargs['data_split_dir'], kwargs['frame_skip'], split='test')
    # relit_dataloader = torch.utils.data.DataLoader(relit_dataset, batch_size=1,
    #                                                shuffle=False, collate_fn=relit_dataset.collate_fn)
    train_sampler = torch.utils.data.distributed.DistributedSampler \
        (relit_dataset, shuffle=False)
    relit_dataloader = torch.utils.data.DataLoader(
        relit_dataset, batch_size=1,
        collate_fn=relit_dataset.collate_fn, sampler=train_sampler)
    total_pixels = relit_dataset.total_pixels
    img_res = relit_dataset.img_res

    # load trained model
    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    print('Loading checkpoint: ', ckpt_path)
    # saved_model_state = torch.load(ckpt_path)
    saved_model_state = torch.load(ckpt_path, map_location={'cuda:2': 'cuda:0','cuda:1': 'cuda:0','cuda:3': 'cuda:0','cuda:4': 'cuda:0','cuda:5': 'cuda:0','cuda:6': 'cuda:0','cuda:7': 'cuda:0'})
    # import ipdb
    # ipdb.set_trace()
    # ## change the lgt of environment for debugging
    # for lgt in saved_model_state["model_state_dict"]['module.envmap_material_network.lgtSGs']:
    #     if lgt[0] < 0 and lgt[1] < 0 and lgt[2] > 0 and lgt[0] > -0.5 and lgt[1] > -0.5:
    #     # if lgt[0] < -0.5 and lgt[1] < -0.5 and lgt[2] > 0.5:
    #         continue
    #     else:
    #         lgt[4:] = lgt[4:] / 100

    # saved_model_state["model_state_dict"]['module.envmap_material_network.specular_reflectance'] += 0.2

    model.load_state_dict(saved_model_state["model_state_dict"])

    print("start render...")
    model.eval()


    images_dir = relitdir
    utils.mkdir_ifnotexists(images_dir)
    print('Output directory is: ', images_dir)

    with open(os.path.join(relitdir, 'ckpt_path.txt'), 'w') as fp:
        fp.write(ckpt_path + '\n')

    relit_with_light(model, relit_dataloader, images_dir, total_pixels,
                     img_res, albedo_ratio=None, light_type='origin',
                     trainstage=trainstage)


    # envmap6_path = './envmaps/envmap6'
    # print('Loading light from: ', envmap6_path)
    # model.module.envmap_material_network.load_light(envmap6_path)
    # images_dir = relitdir + '_envmap6_relit'
    # utils.mkdir_ifnotexists(images_dir)
    # print('Output directory is: ', images_dir)
    # relit_with_light(model, relit_dataloader, images_dir,
    #                 total_pixels, img_res, albedo_ratio=None,
    #                  light_type='envmap6', trainstage=trainstage)
    #
    # envmap12_path = './envmaps/envmap12'
    # print('Loading light from: ', envmap12_path)
    # model.module.envmap_material_network.load_light(envmap12_path)
    # images_dir = relitdir + '_envmap12_relit'
    # utils.mkdir_ifnotexists(images_dir)
    # print('Output directory is: ', images_dir)
    # relit_with_light(model, relit_dataloader, images_dir,
    #                 total_pixels, img_res, albedo_ratio=None,
    #                  light_type='envmap12', trainstage=trainstage)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be relituated.')
    parser.add_argument('--exps_folder', type=str, default='', help='The experiments folder name.')

    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when test')

    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The trained model checkpoint to test')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--if_indirect', default=False, action="store_true",
                        help='If set, render with indirect light.')
    parser.add_argument('--if_silhouette', default=False, action="store_true",
                        help='If set, use reynolds transport.')
    parser.add_argument('--trainstage', type=str, default='IDR', help='')
    parser.add_argument('--if_dist_weight', default=False, action="store_true",
                        help='lgt source weight.')
    parser.add_argument('--unet', default=False, action="store_true",
                        help='use unet for albedo and roughness')
    parser.add_argument('--real_world', default=False, action="store_true",
                        help='use real world dataset?')
    parser.add_argument('--which_real', type=str, default='')
    opt = parser.parse_args()

    # gpu = opt.gpu
    #
    # if (not gpu == 'ignore'):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )

    relight_obj(conf=opt.conf,
                relits_folder_name='relits',
                data_split_dir=opt.data_split_dir,
                expname=opt.expname,
                exps_folder_name=opt.exps_folder,
                timestamp=opt.timestamp,
                checkpoint=opt.checkpoint,
                frame_skip=opt.frame_skip,
                exp_dir=opt.exp_dir,
                local_rank=opt.local_rank,
                if_indirect=opt.if_indirect,
                if_silhouette=opt.if_silhouette,
                trainstage=opt.trainstage,
                if_dist_weight=opt.if_dist_weight,
                unet=opt.unet,
                real_world=opt.real_world,
                which_real=opt.which_real,
                )
