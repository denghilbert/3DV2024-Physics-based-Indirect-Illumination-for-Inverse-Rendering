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


class GeometryTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.idr_epoch')
        self.max_niters = kwargs['max_niters']
        # self.GPU_INDEX = kwargs['gpu_index']
        self.exp_dir = kwargs['exp_dir']
        self.local_rank = kwargs['local_rank']
        self.expname = 'geometry-' + kwargs['expname']
        self.if_indirect = kwargs['if_indirect']
        self.if_silhouette = kwargs['if_silhouette']
        self.real_world = kwargs['real_world']
        self.which_real = kwargs['which_real']

        ################################# If continue ####################################
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.exp_dir, kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join(self.exp_dir, kwargs['exps_folder_name'], self.expname))
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
        utils.mkdir_ifnotexists(os.path.join(self.exp_dir, self.exps_folder_name))
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
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        ################################ create exp_dir ##################################

        print('shell command : {0}'.format(' '.join(sys.argv)))

        ################################ load data ##################################
        print('Loading data ...')
        if self.real_world:
            if self.which_real == 'DTU':
                self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_real_world')) \
                    (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
            elif self.which_real == 'colmap':
                self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_colmap')) \
                    (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        else:
            self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class')) \
                (kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler \
            (self.train_dataset, shuffle=True)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn, sampler=train_sampler)

        print('Finish loading data ...')
        ################################ load data ##################################

        ################################ model construct ##################################
        self.model = utils.get_class(self.conf.get_string('train.model_class')) \
            (conf=self.conf.get_config('model'), if_indirect=self.if_indirect, if_silhouette=self.if_silhouette)
        if torch.cuda.is_available():
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False, find_unused_parameters=True)
        ################################ model construct ##################################

        ################################ loss and optimizer ##################################
        self.loss = utils.get_class(self.conf.get_string('train.loss_class')) \
            (**self.conf.get_config('loss'))

        self.idr_optimizer = torch.optim.Adam(
            list(self.model.module.implicit_network.parameters()) +
            list(self.model.module.rendering_network.parameters()),
            lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.idr_optimizer,
            self.conf.get_list('train.idr_sched_milestones', default=[]),
            gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))
        ################################ loss and optimizer ##################################

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

        self.num_pixels = self.conf.get_int('train.num_pixels_geometry')
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

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter in self.alpha_milestones:
                    self.loss.alpha = self.loss.alpha * self.alpha_factor

                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_outputs = self.model(model_input, trainstage="geometry")
                loss_output = self.loss(model_outputs, ground_truth,
                                        train_idr=False, train_geometry=True)
                loss = loss_output['loss']
                self.idr_optimizer.zero_grad()
                loss.backward()
                self.idr_optimizer.step()

                if self.cur_iter % 50 == 0:
                    print('{0} [{1}] ({2}/{3}): loss = {4}, '
                        'idr_rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, '
                          'idr_psnr = {8}, idr_lr = {9}'
                            .format(self.expname, epoch, data_index, self.n_batches,
                                    loss.item(),
                                    loss_output['idr_rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    loss_output['mask_loss'].item(),
                                    mse2psnr(loss_output['idr_rgb_loss'].item()),
                                    self.idr_scheduler.get_last_lr()[0],
                                    )
                          )
                    self.writer.add_scalar('idr_rgb_loss', loss_output['idr_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('idr_psnr', mse2psnr(loss_output['idr_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('eikonal_loss', loss_output['eikonal_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('mask_loss', loss_output['mask_loss'].item(), self.cur_iter)

                self.cur_iter += 1
                self.idr_scheduler.step()
