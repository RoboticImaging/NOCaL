"""
Helper functions for NOCaL.
Created: 12/05/22
Author: Ryan Griffiths
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import torch


def relativetransform_pose(pose1, pose2):
    """
    Find the relative transformation between two absolute poses.
    :param pose1: First absolute pose
    :param pose2: Second absolute pose
    :return pose: Relative pose in translation, rotation(euler)
    :return pose_mat: Relative pose in matrix form
    """

    pose1_matrix = contruct_transformation_matrix_from_quat(pose1)
    pose2_matrix = contruct_transformation_matrix_from_quat(pose2)
    pose_mat = np.matmul(np.linalg.inv(pose1_matrix), pose2_matrix)

    rotation = Rotation.from_matrix(pose_mat[:3, :3])
    translation = pose_mat[:3, -1]

    pose = np.zeros(6,)
    pose[:3] = translation
    pose[3:] = rotation.as_euler('xyz')
    return pose, pose_mat


def contruct_transformation_matrix_from_quat(pose):
    """
    Converts a pose in quaternians to matrix form
    :param pose: Pose in quaternians
    :return rt: pose in matric form
    """
    r = Rotation.from_quat(pose[3:])
    t = pose[0:3]
    rt = np.eye(4)
    rt[:3, :3] = r.as_matrix()
    rt[:3, -1] = t
    return rt


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def compute_rotation_matrix_from_euler(euler):
    """
    Code From: https://github.com/papagina/RotationContinuity
    Converts a rotation in euler to matrix form
    :param euler: Rotation in euler form
    :return matrix: Rotation in matrix form
    """
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1) 
    s1 = torch.sin(euler[:, 0]).view(batch, 1)
    c2 = torch.cos(euler[:, 2]).view(batch, 1) 
    s2 = torch.sin(euler[:, 2]).view(batch, 1) 
    c3 = torch.cos(euler[:, 1]).view(batch, 1) 
    s3 = torch.sin(euler[:, 1]).view(batch, 1)

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


def compute_rotation_matrix_from_ortho6d(ortho6d):
    """
    Code From: https://github.com/papagina/RotationContinuity
    Converts a rotation in ortho6d to matrix form
    :param euler: Rotation in ortho6d form
    :return matrix: Rotation in matrix form
    """
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v, return_mag=False):
    """
    Code From: https://github.com/papagina/RotationContinuity
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    """
    Code From: https://github.com/papagina/RotationContinuity
    """
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def log_run(writer, loss, epoch):
    """
    Log losses to tensorboard
    :param writer: Tensorboard writer class
    :param loss: Loss values in a dictonary
    :param epoch: The current epoch for the loss values
    :return:
    """
    for key in loss:
        writer.add_scalar(key, loss[key], epoch + 1)


def log_predicted(writer, epoch, predicted1, predicted2, image1, image2, label):
    """
    Log images to tensorboard
    :param writer: Tensorboard writer class
    :param epoch: The current epoch for the loss values
    :param predicted1, predicted2, image1, image2: images to be logged
    :return:
    """
    writer.add_figure(label,
                      images_vis(image1[:,:,:], image2[:,:,:], predicted1[:,:,:], predicted2[:,:,:], grey=False),
                      global_step=epoch)


def images_vis(im1, im2, predicted1, predicted2, grey=False):
    """
    Generate subplot of figures to be logged
    :param im1, im2, predicted1, predicted2: images to be logged
    :return fig: Figure to be logged
    """
    fig = plt.figure(figsize=(6, 8))
    for idx in np.arange(1):
        ax = fig.add_subplot(5, 2, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(im1[idx], one_channel=grey)
        ax.set_title("T", fontsize=10)

        ax = fig.add_subplot(5, 2, idx + 3, xticks=[], yticks=[])
        matplotlib_imshow(predicted1[idx], one_channel=grey)
        ax.set_title("Predicted T", fontsize=10)

        ax = fig.add_subplot(5, 2, idx + 5, xticks=[], yticks=[])
        matplotlib_imshow(im2[idx], one_channel=grey)
        ax.set_title("T+1", fontsize=10)

        ax = fig.add_subplot(5, 2, idx + 7, xticks=[], yticks=[])
        matplotlib_imshow(predicted2[idx], one_channel=grey)
        ax.set_title("Predicted T+1", fontsize=10)

        ax = fig.add_subplot(5, 2, idx + 9, xticks=[], yticks=[])
        matplotlib_imshow(abs(predicted2[idx]-im2[idx]), one_channel=grey)
        ax.set_title("Difference", fontsize=10)
    return fig


def matplotlib_imshow(img, one_channel=False, showIm=False):
    """
    Convert tensor to figures
    :param img: Image to be plotted
    :param one_channel: If image is single channel (greyscale)
    :param showIm: Plot the image after generating
    :return pos: Generated figure
    """
    img = img.squeeze()
    npimg = img.cpu().detach().numpy()
    npimg = np.clip(npimg, 0, 1)
    if one_channel:
        pos = plt.imshow(npimg, cmap='gray')
        plt.colorbar()
    else:
        pos = plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if showIm:
        plt.show()
    return pos
