import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.sg_envmap_material import EnvmapMaterialNetwork, fibonacci_sphere
from model.sg_render import render_envmap, render_with_all_sg

TINY_NUMBER = 1e-6


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            print('Applying positional encoding to view directions: ', multires_view)
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x = self.tanh(x)
        # return (x + 1.) / 2.
        return x


class LgtCombNet(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, dims=[128, 128, 128, 128]):
        super().__init__()

        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = get_embedder(points_multires)

        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = get_embedder(dirs_multires)

        self.actv_fn = nn.ReLU()

        comb_layer = []
        dim = p_input_dim + dir_input_dim
        for i in range(len(dims)):
            comb_layer.append(nn.Linear(dim, dims[i]))
            comb_layer.append(self.actv_fn)
            dim = dims[i]
        comb_layer.append(nn.Linear(dim, 128))
        self.comb_layer = nn.Sequential(*comb_layer)
        self.sig = nn.Sigmoid()

    def forward(self, points, lgt_dirs):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            lgt_dirs = self.dir_embed_fn(lgt_dirs)

        comb = self.sig(self.comb_layer(torch.cat([points, lgt_dirs], -1)))

        return comb


class IDRNetwork(nn.Module):
    def __init__(self, conf, if_indirect, if_silhouette, if_comb=False, if_dist_weight=False, unet=False):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))

        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.lgt_combination_network = LgtCombNet(**conf.get_config('lgt_combination_network'))
        self.envmap_material_network = EnvmapMaterialNetwork(**conf.get_config('envmap_material_network'))


        self.if_indirect = if_indirect
        self.if_silhouette = if_silhouette
        self.if_comb = if_comb
        self.if_dist_weight = if_dist_weight
        self.unet = unet

    def forward(self, input, trainstage='geometry', if_testing=False):
        # parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(
                sdf=lambda x: self.implicit_network(x)[:, 0], cam_loc=cam_loc,
                object_mask=object_mask, ray_directions=ray_dirs)

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
        sdf_output = self.implicit_network(points)[:, 0:1]

        ray_dirs = ray_dirs.reshape(-1, 3)

        return_obj = {
            'points': points,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
        }


        if trainstage == 'geometry':
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            diff_surface_points = self.sample_network(surface_output,
                                                      surface_sdf_values,
                                                      surface_points_grad,
                                                      surface_dists,
                                                      surface_cam_loc,
                                                      surface_ray_dirs)
            view = -ray_dirs[surface_mask]
            normals = torch.ones_like(points).float().cuda()
            idr_rgb =  torch.ones_like(points).float().cuda()
            if diff_surface_points.shape[0] > 0:
                normals[surface_mask], idr_rgb[surface_mask] = self.get_idr_render(diff_surface_points, view)
            return_obj.update({ 'idr_rgb': idr_rgb,
                                'normals': normals,
                                'grad_theta': grad_theta})

            return return_obj

        ### train DIP networks
        if trainstage == 'DIP' and if_testing == False:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            diff_surface_points = self.sample_network(surface_output,
                                                    surface_sdf_values,
                                                    surface_points_grad,
                                                    surface_dists,
                                                    surface_cam_loc,
                                                    surface_ray_dirs)


        if if_testing == True:
            surface_mask = network_object_mask
            diff_surface_points = points[surface_mask]


        bg_rgb_values = torch.ones_like(points).float().cuda()
        if self.envmap_material_network.envmap is not None:
            bg_rgb_values = render_envmap(self.envmap_material_network.envmap, ray_dirs)

        ### peform sg render for materials training
        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        indir_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_specular_rgb_values = torch.ones_like(points).float().cuda()

        normal_values = torch.ones_like(points).float().cuda()
        diffuse_albedo_values = torch.ones_like(points).float().cuda()
        roughness_values = torch.ones_like(points).float().cuda()

        vis_shadow = torch.ones_like(points).float().cuda()

        random_xi_diffuse_albedo = torch.ones_like(points).float().cuda()
        random_xi_roughness = torch.ones_like(points).float().cuda()
        kl_output = torch.ones_like(points).float().cuda()


        albedo_ratio = None
        if 'albedo_ratio' in input:
            albedo_ratio = input['albedo_ratio']

        if diff_surface_points.shape[0] > 0:
            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            mask_indir_lgs = None
            ret = self.get_sg_render(diff_surface_points,
                                     view_dirs, 
                                     mask_indir_lgs,
                                     albedo_ratio=albedo_ratio,
                                     if_testing=if_testing)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            indir_rgb_values[surface_mask] = ret['indir_rgb']
            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

            normal_values[surface_mask] = ret['normals']
            diffuse_albedo_values[surface_mask] = ret['diffuse_albedo']
            roughness_values[surface_mask] = ret['roughness'].expand(-1, 3)

            vis_shadow[surface_mask] = ret['vis_shadow']

            random_xi_diffuse_albedo[surface_mask] = ret['random_xi_diffuse_albedo']
            random_xi_roughness[surface_mask] = ret['random_xi_roughness'].expand(-1, 3)
            kl_output = ret['kl_output']
        else:
            ret = {'comb': None, 'angle_reflectnormal_128': None, 'sg_specular_reflectance': torch.tensor([0.02]).cuda()}

        if not if_testing:
            return_obj.update({
                'grad_theta': grad_theta,
                'idr_rgb': idr_rgb_values,
                'normals': normal_values,

                'bg_rgb': bg_rgb_values,
                'sg_rgb': sg_rgb_values,
                'indir_rgb': indir_rgb_values,
                'sg_diffuse_rgb': sg_diffuse_rgb_values,
                'sg_specular_rgb': sg_specular_rgb_values,
                'diffuse_albedo': diffuse_albedo_values,
                'roughness': roughness_values,
                'vis_shadow': vis_shadow,
                'random_xi_roughness': random_xi_roughness,
                'random_xi_diffuse_albedo': random_xi_diffuse_albedo,
                'kl_output': kl_output,

                'comb': ret['comb'],
                'angle_reflectnormal_128': ret['angle_reflectnormal_128'],
                'sg_specular_reflectance': ret['sg_specular_reflectance'],
                'sharpness': ret['sharpness'],
                'amptitude': ret['amptitude'],
            })
        else:
            return_obj.update({
                'normals': normal_values,
                'bg_rgb': bg_rgb_values,
                'sg_rgb': sg_rgb_values,
                'indir_rgb': indir_rgb_values,
                'sg_diffuse_rgb': sg_diffuse_rgb_values,
                'sg_specular_rgb': sg_specular_rgb_values,
                'diffuse_albedo': diffuse_albedo_values,
                'roughness': roughness_values,
            })

        return return_obj
    
    def get_idr_render(self, points, view_dirs=None, normal_only=False):
        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        if normal_only:
            return normals

        # idr renderer
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        return normals, idr_rgb      

    def get_sg_render(self, points, view_dirs, indir_lgtSGs, albedo_ratio=None, if_testing=False):
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        normals, idr_rgb = self.get_idr_render(points, view_dirs)
        ret = { 'normals': normals, 'idr_rgb': idr_rgb}


        # sg renderer
        sg_envmap_material = self.envmap_material_network(points, unet=self.unet)

        if albedo_ratio is not None:
            sg_envmap_material['sg_diffuse_albedo'] = sg_envmap_material['sg_diffuse_albedo'] * albedo_ratio

        sg_ret = render_with_all_sg(points=points.detach(),
                                    normal=normals.detach(), 
                                    viewdirs=view_dirs, 
                                    lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                    indir_lgtSGs=None,
                                    VisModel=None,
                                    comp_vis=False,
                                    ray_tracer=self.ray_tracer,
                                    implicit_network=self.implicit_network,
                                    fibonacci_sphere=self.envmap_material_network.fibonacci_sphere,
                                    if_indirect=self.if_indirect,
                                    if_silhouette=self.if_silhouette,
                                    lgt_combination_network=self.lgt_combination_network,
                                    if_testing=if_testing,
                                    if_comb=self.if_comb,
                                    if_dist_weight=self.if_dist_weight
                                    )

        ret.update(sg_ret)
        ret.update({'diffuse_albedo': sg_envmap_material['sg_diffuse_albedo'],
                    'roughness': sg_envmap_material['sg_roughness'],
                    'random_xi_roughness': sg_envmap_material['random_xi_roughness'],
                    'random_xi_diffuse_albedo': sg_envmap_material['random_xi_diffuse_albedo'],
                    'kl_output': sg_envmap_material['kl_output'],
                    'sg_specular_reflectance': sg_envmap_material['sg_specular_reflectance'],
        })

        return ret
    
    def batch_idr_forward(self, points, viewdirs):
        batch_size = 20000
        radiance = torch.zeros_like(points).cuda()
        split = []

        for i, indx in enumerate(torch.split(torch.arange(points.shape[0]).cuda(), batch_size, dim=0)):
            curr_points = points[indx]
            curr_viewdirs = viewdirs[indx]
            feature_vectors = None
            if self.feature_vector_size > 0:
                feature_vectors = self.implicit_network(curr_points)[:, 1:]
            g = self.implicit_network.gradient(curr_points).clone().detach()
            normals = g[:, 0, :] / (torch.norm(g[:, 0, :], dim=-1, keepdim=True) + TINY_NUMBER)
            radiance[indx] = self.rendering_network(curr_points, normals, curr_viewdirs, feature_vectors)

        return radiance.detach()

    def sample_dirs(self, normals, r_theta, r_phi):
        z_axis = torch.zeros_like(normals).cuda()
        z_axis[:, :, 0] = 1

        def norm_axis(x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)

        normals = norm_axis(normals)
        U = norm_axis(torch.cross(z_axis, normals))
        V = norm_axis(torch.cross(normals, U))

        r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
        r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)
        sample_raydirs = U * torch.cos(r_theta) * torch.sin(r_phi) \
                        + V * torch.sin(r_theta) * torch.sin(r_phi) \
                        + normals * torch.cos(r_phi) # [num_cam, num_samples, 3]
        return sample_raydirs