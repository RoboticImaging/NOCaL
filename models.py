"""
Models for running pose est and rendering
Created: 28/02/2022
Author: Ryan Griffiths
"""

import torch
import torch.nn as nn
from torchmeta import custom_layers, hyper_net
import numpy as np
import geometry
import util


class NOCaL(nn.Module):
    """
    NOCaL Framework
    """
    def __init__(self, latent_dim, num_hidden_units_phi, parameterization='plucker', depth=False, enc_freq=4, image_size=[40,30]):
        super(NOCaL, self).__init__()
        self.parameterization = parameterization
        self.depth = depth

        in_channels = 6
        out_dim = 3
        in_dim = 6

        # Select the parameterization for the rays
        if parameterization == 'planes':
            in_dim -= 2
            self.ray_params = geometry.planes_embedding
        elif parameterization == 'plucker':
            self.ray_params = geometry.plucker_embedding
        elif parameterization == 'plenoptic':
            self.ray_params = geometry.plenoptic_embedding

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = num_hidden_units_phi

        self.latent_model = Encoder(in_channels=in_channels, latent_size=latent_dim)

        self.encoder, encoder_dim = get_embedder(multires=enc_freq, in_dim=in_dim)
        in_dim = encoder_dim

        # Define LFN
        self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=6,
                                         in_features=in_dim, out_features=out_dim, outermost_linear=True,
                                         norm='layernorm_na')

        # Define Hypernetwork
        self.hyper_model = hyper_net.HyperNetwork(hyper_in_features=self.latent_dim,
                                                  hyper_hidden_layers=1,
                                                  hyper_hidden_features=self.latent_dim,
                                                  hypo_module=self.phi)

        # Define Distortion model
        self.distortion_model = self.MLP(in_features=2, out_features=2, hidden_units=8, layers=4)
        self.distortion_model.apply(self.init_weights_normal)

        pose1 = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
        self.pose1 = torch.from_numpy(pose1).float()


    def MLP(self, in_features, out_features, hidden_units, layers):
        mlp = [nn.Linear(in_features, hidden_units)]
        for i in range(layers-1):
            mlp.append(nn.Linear(hidden_units, hidden_units))
            mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(hidden_units, out_features))
        return nn.Sequential(*mlp)

    # Initialise weights using xavier uniform
    def init_weights_normal(self, m):
        if isinstance(m, nn.Linear):
            m.bias.data.fill_(0.002)
            torch.nn.init.xavier_uniform_(m.weight, gain=0.05)

    def forward(self, im1, im2, pose, uv, k, labelled, distort):
        b, c, m, n = im1.size()
        encoding, pose_est = self.latent_model(im1, im2)

        phi_weights = self.hyper_model(encoding)
        phi_loaded = lambda x: self.phi(x, params=phi_weights)

        if distort:
            uv_distort = uv.clone()
            uv_distort[:, :, 0] = (uv_distort[:, :, 0] - torch.max(uv_distort[:,:,0]))/torch.max(uv_distort[:,:,0])
            uv_distort[:, :, 1] = (uv_distort[:, :, 1] - torch.max(uv_distort[:,:,1]))/torch.max(uv_distort[:,:,1])
            camera_space_dist = self.distortion_model(uv_distort)
        else:
            camera_space_dist = None

        im1_rays = self.ray_params(self.pose1.repeat(b,1,1).to(im1.device), uv, k, camera_space_dist)

        if labelled:
            im2_rays = self.ray_params(pose, uv, k, camera_space_dist)
        else:
            im2_rays = self.ray_params(pose_est, uv, k, camera_space_dist)

        im1_rays = self.encoder(im1_rays)
        im2_rays = self.encoder(im2_rays)

        predicted_1 = phi_loaded(im1_rays)
        predicted_2 = phi_loaded(im2_rays)

        im1_predicted = predicted_1[:, :, :].permute((0,2,1)).view(b, c, m, n)
        im2_predicted = predicted_2[:, :, :].permute((0,2,1)).view(b, c, m, n)

        predicted_images = (im1_predicted, im2_predicted)

        return predicted_images, pose_est, encoding, camera_space_dist


class CameraParams(nn.Module):
    """
    Camera Parameter NetNetwork used to estimate intrinsics
    """
    def __init__(self, init_focal=1.0, image_size=[80,60]):
        super(CameraParams, self).__init__()
        focal_tensor = torch.tensor(init_focal)
        self.f_x = nn.Parameter(focal_tensor, requires_grad=True)
        self.f_y = nn.Parameter(focal_tensor, requires_grad=True)
        self.image_size = image_size

    def forward(self):
        k = torch.Tensor(
                [[self.f_x, 0., self.image_size[0]/2, 0.],
                 [0., self.f_y, self.image_size[1]/2, 0],
                 [0., 0, 1, 0],
                 [0, 0, 0, 1]])
        return k


class Encoder(nn.Module):
    """
    Encoder for image pairs.
    Code originially from: https://github.com/RoboticImaging/LearnLFOdo_IROS2021
    """
    def __init__(self, in_channels=3, nb_ref_imgs=1, latent_size=128):
        super(Encoder, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        def conv_pn(in_planes, out_planes, kernel_size=3):
            """
            Convolutional Layer followed by ReLU
            :param in_planes: number of channels in the input
            :type in_planes: int
            :param out_planes: number of channels in the output
            :type out_planes: int
            :param kernel_size: Size of the convolving kernel
            :type kernel_size: int or tuple
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2,
                          stride=2),
                nn.LeakyReLU(inplace=True)
            )

        # define the convolutional + ReLU layers [default kernel_size=3, stride=2, padding = (kernel_size -1)//2]
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_pn(in_channels, conv_planes[0], kernel_size=7)
        self.conv2 = conv_pn(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv_pn(conv_planes[1], conv_planes[2])
        self.conv4 = conv_pn(conv_planes[2], conv_planes[3])
        self.conv5 = conv_pn(conv_planes[3], conv_planes[4])
        self.conv6 = conv_pn(conv_planes[4], conv_planes[5])
        self.conv7 = conv_pn(conv_planes[5], conv_planes[6])

        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
        )

        # final layer is a 1x1 convolutional layer
        self.pose_pred = nn.Sequential(
            nn.Conv2d(conv_planes[6], 9 * self.nb_ref_imgs, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(9 * self.nb_ref_imgs, 9 * self.nb_ref_imgs, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(9 * self.nb_ref_imgs, 9 * self.nb_ref_imgs, kernel_size=1, padding=0),

        )
        # self.pose_pred = nn.Conv2d(conv_planes[6], 9 * self.nb_ref_imgs, kernel_size=1, padding=0)
        self.latent_out = nn.Conv2d(conv_planes[6], latent_size, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def init_weights(self):
        """
        Initializes weights with Xavier Uniform weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, tgt_formatted, tgt_unformatted, ref_formatted, ref_unformatted,
               tgt_formatted_ex=None, ref_formatted_ex=None):
        """
        Encodes the lightfield images
        :param tgt_formatted: the target tiled epipolar image [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
        :type tgt_formatted: tensor
        :param tgt_unformatted: the target grid of images stacked on the colour-channel   [B, N, H, W]
        :type tgt_unformatted: tensor
        :param ref_formatted: list of reference tiled epipolar image [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
        :type ref_formatted: tensor
        :param ref_unformatted: list of grid of images stacked on the colour-channel   [B, N, H, W]
        :type ref_unformatted: tensor
        :return: the encoded target image concatenated with the stacked image-grid   [B, N+16, H, W],
         the same for each of the images of the list of reference images
        :rtype: tuple of tensor, list of tensors
        """
        return self.encoder(tgt_formatted, tgt_unformatted, ref_formatted, ref_unformatted,
                            tgt_formatted_ex, ref_formatted_ex)


    def forward(self, target_image: torch.Tensor, ref_imgs: torch.Tensor, rev=False):
        """
        Forward pass of the network
        :param target_image: Target image   [B, channels, h, w]
        :type target_image: tensor
        :param ref_imgs: List of reference images   list of images of shape [B, channels, h, w]
        :type ref_imgs: list of tensors
        :return: the 6DOF pose of the target frame relative to the reference frames
        :rtype: tensor
        """

        i_input = torch.cat((target_image, ref_imgs), dim=1)  # convert it to a tensor
        out_conv1 = self.conv1(i_input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        latent = self.latent_out(out_conv7)
        latent = self.relu(latent)
        latent = latent.mean(3).mean(2)
        latent = latent.view(latent.size(0), -1)
        latent = self.fc(latent)
        latent = self.relu(latent)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), 9)
        rotation = util.compute_rotation_matrix_from_ortho6d(pose[:, 3:])
        pose_mat = torch.cat((rotation, pose[..., :3].unsqueeze(2)), dim=2)
        pose_mat = torch.cat((pose_mat, torch.tensor([0, 0, 0, 1]).repeat(pose.size(0), 1).unsqueeze(1).to(target_image.device)), dim=1)
        return latent, pose_mat


class Embedder:
    """
    Positional encoding as used in the original NeRF, using sine/cosine encodings
    https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    """
    def __init__(self, **kwargs):
        self.out_dim = None
        self.embed_fns = None
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, in_dim, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': in_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
