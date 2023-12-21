import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt




class DIPTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.material_epoch')
        self.max_niters = kwargs['max_niters']
        self.exp_dir = kwargs['exp_dir']
        self.local_rank = kwargs['local_rank']
        self.expname = 'after_geometry-' + kwargs['expname']
        if 'hot' in kwargs['expname']:
            self.last_expname = 'geometry-hot'
        elif 'air' in kwargs['expname']:
            self.last_expname = 'geometry-air'
        elif 'jug' in kwargs['expname']:
            self.last_expname = 'geometry-jug'
        elif 'cha' in kwargs['expname']:
            self.last_expname = 'geometry-cha'
            self.nepochs = self.nepochs / 2
        else:
            self.last_expname = 'geometry-' + kwargs['expname']

        # ## this is for ablations on one scene
        self.if_indirect = kwargs['if_indirect']
        self.if_silhouette = kwargs['if_silhouette']
        self.if_comb = kwargs['if_comb']
        self.if_dist_weight = kwargs['if_dist_weight']
        self.unet = kwargs['unet']
        self.real_world = kwargs['real_world']
        ################################# If continue ####################################
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.exp_dir,kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join(self.exp_dir,kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']
        ################################# If continue ####################################

        ################################ create exp_dir ##################################
        utils.mkdir_ifnotexists(os.path.join(self.exp_dir,self.exps_folder_name))
        self.expdir = os.path.join(self.exp_dir, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"
        self.comb_optimizer_params_subdir = "COMBOptimizerParameters"
        self.comb_scheduler_params_subdir = "COMBSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.comb_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.comb_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        ################################ create exp_dir ##################################

        print('shell command : {0}'.format(' '.join(sys.argv)))

        ################################ load data ##################################
        print('Loading data ...')
        if self.real_world:
            self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_real_world')) \
                (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        else:
            self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))\
                    (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler\
            (self.train_dataset, shuffle=True)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn, sampler = train_sampler)

        print('Finish loading data ...')
        ################################ load data ##################################

        ################################ model construct ##################################
        self.model = utils.get_class(self.conf.get_string('train.model_class'))\
            (conf=self.conf.get_config('model'), if_indirect=self.if_indirect,
             if_silhouette=self.if_silhouette, if_comb=self.if_comb,
             if_dist_weight=self.if_dist_weight, unet=self.unet)
        if torch.cuda.is_available():
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False, find_unused_parameters=True)
        ################################ model construct ##################################

        ################################ loss and optimizer ##################################
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))\
            (**self.conf.get_config('loss'))

        self.idr_optimizer = torch.optim.Adam(
            list(self.model.module.implicit_network.parameters()) +
            list(self.model.module.rendering_network.parameters()),
            lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.idr_optimizer,
            self.conf.get_list('train.idr_sched_milestones', default=[]),
            gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))

        self.sg_optimizer = torch.optim.Adam(
            self.model.module.envmap_material_network.parameters(),
            lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.sg_optimizer,self.conf.get_list('train.sg_sched_milestones',default=[]),
            gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))

        self.comb_optimizer = torch.optim.Adam(
            self.model.module.lgt_combination_network.parameters(),
            lr=self.conf.get_float('train.comb_learning_rate'))
        self.comb_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.comb_optimizer, self.conf.get_list('train.comb_sched_milestones', default=[]),
            gamma=self.conf.get_float('train.comb_sched_factor', default=0.0))
        ################################ loss and optimizer ##################################

        ################################ loading last check ##################################
        if not is_continue:
            last_timestamps = os.listdir(os.path.join(self.exp_dir, kwargs['exps_folder_name'], self.last_expname))
            last_timestamps = sorted(last_timestamps)[-1]
            last_dir = os.path.join(self.exp_dir, self.exps_folder_name, self.last_expname)
            old_checkpnts_dir = os.path.join(last_dir, last_timestamps, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
            )
            del saved_model_state["model_state_dict"]['module.envmap_material_network.lgtSGs']
            del saved_model_state["model_state_dict"]['module.lgt_combination_network.comb_layer.8.bias']
            del saved_model_state["model_state_dict"]['module.lgt_combination_network.comb_layer.8.weight']
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            self.start_epoch = saved_model_state['epoch']
            self.start_epoch = 0

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_scheduler.load_state_dict(data["scheduler_state_dict"])
        ################################ loading last check ##################################

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            print('Loading pretrained model: ', os.path.join(
                old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth")
            )
            self.model.load_state_dict(saved_model_state["model_state_dict"])   
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_scheduler.load_state_dict(data["scheduler_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_scheduler.load_state_dict(data["scheduler_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.comb_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.comb_optimizer.load_state_dict(data["optimizer_state_dict"])
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.comb_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.comb_scheduler.load_state_dict(data["scheduler_state_dict"])


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch * self.n_batches > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.comb_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.comb_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.comb_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.comb_optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.comb_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.comb_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.comb_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.comb_scheduler_params_subdir, "latest.pth"))



    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        for param in self.model.module.implicit_network.parameters():
            param.requires_grad = False
        for param in self.model.module.rendering_network.parameters():
            param.requires_grad = False


        for epoch in range(self.start_epoch, int(self.nepochs) + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter in self.alpha_milestones:
                    self.loss.alpha = self.loss.alpha * self.alpha_factor

                if epoch % 20 == 0:
                    self.save_checkpoints(epoch)

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_outputs = self.model(model_input, trainstage="DIP")
                loss_output = self.loss(
                    model_outputs, ground_truth,
                    mat_model=self.model.module.envmap_material_network,
                    train_idr=True, train_geometry=False)

                loss = loss_output['loss']

                self.idr_optimizer.zero_grad()
                self.sg_optimizer.zero_grad()
                self.comb_optimizer.zero_grad()
                loss.backward()
                self.idr_optimizer.step()
                self.sg_optimizer.step()
                self.comb_optimizer.step()


                if self.cur_iter % 50 == 0:
                    print('{0} [{1}] ({2}/{3}): loss = {4}, '
                        'idr_rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, '
                          'idr_psnr = {8}, idr_lr = {9}, sg_rgb_loss = {10}, '
                          'latent_smooth_loss = {11}, kl_loss = {12}, comb_loss = {13}, '
                          'sg_psnr={14}, sg_specular_reflectance={15}, rough_regu={16},'
                          'sharpness={17}, amptitude={18}'
                            .format(self.expname, epoch, data_index, self.n_batches, 
                                    loss.item(),
                                    loss_output['idr_rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    loss_output['mask_loss'].item(),
                                    mse2psnr(loss_output['idr_rgb_loss'].item()),
                                    self.idr_scheduler.get_last_lr()[0],
                                    loss_output['sg_rgb_loss'].item(),
                                    loss_output['latent_smooth_loss'].item(),
                                    loss_output['kl_loss'].item(),
                                    loss_output['comb_loss'].item(),
                                    mse2psnr(loss_output['sg_rgb_loss'].item()),
                                    loss_output['sg_specular_reflectance'].item(),
                                    loss_output['rough_regu'].item(),
                                    loss_output['sharpness'].item(),
                                    loss_output['amptitude'].item(),
                                    ))
                    self.writer.add_scalar('idr_rgb_loss', loss_output['idr_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('idr_psnr', mse2psnr(loss_output['idr_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('eikonal_loss', loss_output['eikonal_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('mask_loss', loss_output['mask_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('latent_smooth_loss', loss_output['latent_smooth_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('kl_loss', loss_output['kl_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('comb_loss', loss_output['comb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('sg_specular_reflectance', loss_output['sg_specular_reflectance'].item(), self.cur_iter)
                    self.writer.add_scalar('rough_regu', loss_output['rough_regu'].item(), self.cur_iter)
                    self.writer.add_scalar('sharpness', loss_output['sharpness'].item(), self.cur_iter)
                    self.writer.add_scalar('amptitude', loss_output['amptitude'].item(), self.cur_iter)


                self.cur_iter += 1
                self.idr_scheduler.step()
                self.sg_scheduler.step()
                self.comb_scheduler.step()