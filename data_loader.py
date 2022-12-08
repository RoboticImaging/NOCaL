"""
Load and process data
Created: 25/02/22
Author: Ryan Griffiths
"""

import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
import numpy as np
from skimage.io import imread
import natsort
import glob
from pandas import read_csv
import cv2
import util


# General Dataset class for own data
class ImageDataset(Dataset):
    # load the dataset
    def __init__(self, path, im_size, scenes, steps, grey_scale=False):
        # load the csv file as a dataframe
        self.scene_names = scenes
        self.orig_size = [640, 480]
        self.traj_steps = steps
        self.im_size = im_size
        self.path = path
        random.seed(10)

        self.scenes_img = []
        self.scenes_pose = []
        self.scenes_k = []
        self.scenes_uv = []

        for i in self.scene_names:
            img_files = glob.glob(path + '/' + i + '/cam0/*.csv')
            img_files = read_csv(img_files[0])

            image_list = []

            for id in range(len(img_files)):
                image = torch.from_numpy(cv2.resize(
                    np.array(imread((path + '/' + i + '/cam0/data/' + img_files.iloc[id, 1]), as_gray=grey_scale).astype('float32'))/255.0,
                    (im_size[0], im_size[1]), interpolation=cv2.INTER_AREA))
                if grey_scale:
                    image = torch.unsqueeze(image, 0)
                else:
                    image = image.permute((2, 0, 1))
                image_list.append(image)

            self.scenes_img.append(image_list)

            pose_path = glob.glob(path + '/' + i + '/*.gt')
            self.scenes_pose.append(read_csv(pose_path[0]))

            k = np.loadtxt(path + '/' + i + '/cameraInfo.txt')
            full_intrinsic = np.array(
                [[k[0, 0], 0., k[0, 2], 0.],
                 [0., k[1, 1], k[1, 2], 0],
                 [0., 0, 1, 0],
                 [0, 0, 0, 1]])
            intrinsics = torch.from_numpy(full_intrinsic).float()
            self.scenes_k.append(intrinsics)

            uv = np.mgrid[0:self.orig_size[1], 0:self.orig_size[0]].astype(np.int32).transpose(1, 2, 0)
            uv = cv2.resize(uv, (im_size[0], im_size[1]), interpolation=cv2.INTER_NEAREST)
            uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
            self.scenes_uv.append(uv.reshape(-1, 2).float())

    # number of rows in the dataset
    def __len__(self):
        length = 0
        for i in range(len(self.scenes_img)):
            length += len(self.scenes_img[i])

        if self.threeimages:
            return length - 3*len(self.scenes_img)
        else:
            return length - self.traj_steps*len(self.scenes_img)

    # get a row at an index
    def __getitem__(self, idx):
        image_offset = random.randint(1, self.traj_steps)
        scene = 0
        for i in range(len(self.scenes_img)):
            if idx < len(self.scenes_img[i]) - self.traj_steps:
                scene = i
                break
            else:
                idx -= len(self.scenes_img[i]) - self.traj_steps

        pose1_quat = np.array(
            [-self.scenes_pose[scene]['y'][idx], self.scenes_pose[scene]['x'][idx], self.scenes_pose[scene]['z'][idx],
             self.scenes_pose[scene]['q_x'][idx], -self.scenes_pose[scene]['q_y'][idx], -self.scenes_pose[scene]['q_z'][idx], self.scenes_pose[scene]['q_w'][idx]])
        pose2_quat = np.array(
            [-self.scenes_pose[scene]['y'][idx + image_offset], self.scenes_pose[scene]['x'][idx + image_offset], self.scenes_pose[scene]['z'][idx + image_offset],
             self.scenes_pose[scene]['q_x'][idx + image_offset], -self.scenes_pose[scene]['q_y'][idx + image_offset],
             -self.scenes_pose[scene]['q_z'][idx + image_offset], self.scenes_pose[scene]['q_w'][idx + image_offset]])

        # Randomly decide which way the frames are inputed, then calcualte transform between them
        inverse = random.randint(0,1)
        if inverse:
            _, pose1 = util.relativetransform_pose(pose2_quat, pose1_quat)
        else:
            _, pose1 = util.relativetransform_pose(pose1_quat, pose2_quat)

        pose1 = torch.from_numpy(pose1).float()

        if inverse:
            images = (self.scenes_img[scene][idx + image_offset],
                    self.scenes_img[scene][idx])
        else:
            images = (self.scenes_img[scene][idx],
                        self.scenes_img[scene][idx + image_offset])

        return images, pose1, self.scenes_uv[scene], self.scenes_k[scene]

    def read_image(self, path):
        image = imread(path, as_gray=True).astype('float32')
        image = cv2.resize(np.array(image), (self.im_size[0], self.im_size[1]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image)

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * (self.__len__()))
        train_size = self.__len__() - test_size
        # calculate the split
        return random_split(self, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    def dataset_num(self, num):
        train, _ = random_split(self, [num, self.__len__() - num])
        return train


# Dataset class for the Light Field odometry dataset
class ImageDatasetLFOdo(Dataset):
    # load the dataset
    def __init__(self, path, im_size, mode='train', steps=1):
        self.scenes_img = []
        self.scenes_pose = []
        self.traj_steps = steps

        # load the csv file as a dataframe
        with open(path+'/core/'+mode+'.txt') as f:
            scenes = f.readlines()

        img_files = glob.glob(path + '/core/' + scenes[0].rstrip() + '/0/*.png')
        orig_size = imread(img_files[0]).shape

        for seq in scenes:
            img_files = glob.glob(path + '/core/' + seq.rstrip() + '/0/*.png')
            img_files = natsort.natsorted(img_files)

            image_list = []
            for img_path in img_files:
                image = torch.from_numpy(cv2.resize(
                    np.array(
                        imread(img_path, as_gray=False).astype('float32')) / 255.0,
                        (im_size[0], im_size[1]), interpolation=cv2.INTER_AREA))
                image = image.permute((2, 0, 1))
                image_list.append(image)
            self.scenes_img.append(image_list)

            pose_gt = np.load(path+'/core/'+seq.rstrip()+'/poses_gt_prev_curr_cam.npy')
            self.scenes_pose.append(pose_gt)

        k = np.loadtxt(path + '/cameraInfo.txt')
        full_intrinsic = np.array(
            [[k[0, 0], 0., k[0, 2], 0.],
             [0., k[1, 1], k[1, 2], 0],
             [0., 0, 1, 0],
             [0, 0, 0, 1]])
        intrinsics = torch.from_numpy(full_intrinsic).float()
        self.k = intrinsics

        # Produce the uv values for the camera
        uv = np.mgrid[0:orig_size[0], 0:orig_size[1]].astype(np.int32).transpose(1, 2, 0)
        uv = cv2.resize(uv, (im_size[0], im_size[1]), interpolation=cv2.INTER_NEAREST)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        self.uv = uv.reshape(-1, 2).float()

    # number of rows in the dataset
    def __len__(self):
        length = 0
        for i in range(len(self.scenes_img)):
            length += len(self.scenes_img[i])

        return length - self.traj_steps * len(self.scenes_img)

    # get a row at an index
    def __getitem__(self, idx):
        scene = 0
        for i in range(len(self.scenes_img)):
            if idx < len(self.scenes_img[i]) - self.traj_steps:
                scene = i
                break
            else:
                idx -= len(self.scenes_img[i]) - self.traj_steps

        image_offset = random.randint(1, self.traj_steps)

        image1 = self.scenes_img[scene][idx]
        image2 = self.scenes_img[scene][idx + image_offset]

        images = (image1, image2)

        pose = self.scenes_pose[scene][idx + 1]
        for i in range(image_offset-1):
            pose = np.matmul(pose, self.scenes_pose[scene][idx+i+2])

        inverse = random.randint(0, 0)

        if inverse:
            images = tuple(reversed(images))
            rotation = pose[:3,:3]
            pose = np.concatenate((np.concatenate([rotation, np.expand_dims(np.matmul(-rotation.T, pose[:3, 3]), 1)], 1), [[0, 0, 0, 1]]))

        return images, torch.from_numpy(pose).float(), self.uv, self.k


# Load the data and produce dataloaders for the given config file
def prepare_data(args):
    if args.dataset == 'LFOdo':
        train = ImageDatasetLFOdo(args.data_path, args.im_size, mode='train', steps=args.steps)
        test = ImageDatasetLFOdo(args.data_path, args.im_size, mode='val', steps=1)
    else:
        train = ImageDataset(args.path, args.im_size, args.train_scenes, steps=args.steps)
        test = ImageDataset(args.path, args.im_size, args.test_scenes, steps=1)

    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_dl, test_dl
