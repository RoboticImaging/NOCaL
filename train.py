"""
NOCaL: implicit rendering, camera parameter and pose estimation
Main script to run training.
Created: 25/02/22
Author: Ryan Griffiths
"""


import data_loader
import models
import util
import pytorch_ssim
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import configargparse
from datetime import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    print("Running on " + str(device))

    # Load arguments from config file
    parser = config_parser()
    args = parser.parse_args()

    args.save_path = '{}_NOCaL'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_writer = SummaryWriter('tensorboard_data/{}{}'.format(args.save_folder, args.save_path))

    # Load data
    print("Loading Data")
    train_dl, test_dl = data_loader.prepare_data(args)

    # Load model
    model = models.NOCaL(args.latent_size, args.hidden_units, args.parameterization,
                                         enc_freq=args.enc_freq, image_size=args.im_size)
    cam_param = models.CameraParams(init_focal=args.focal_length)

    model = model.to(device)
    cam_param = cam_param.to(device)

    # Train network with given config
    print("Training Model")
    train(args, model, cam_param, train_dl, test_dl, log_writer)


# Parse the Config file
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_scenes', type=str, nargs='*')
    parser.add_argument('--test_scenes', type=list, nargs='*')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--encoder_lr', type=float)
    parser.add_argument('--hyper_lr', type=float)
    parser.add_argument('--distortion_lr', type=float)
    parser.add_argument('--intrinsics_lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--latent_size', type=int)
    parser.add_argument('--hidden_units', type=int)
    parser.add_argument('--parameterization', type=str)
    parser.add_argument('--enc_freq', type=int)
    parser.add_argument('--label_ratio', type=float)
    parser.add_argument('--im_size', type=int, nargs='*')
    parser.add_argument('--focal_length', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--distortion_start', type=int)
    parser.add_argument('--focal_start', type=int)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--checkpoint_period', type=int)
    return parser


# A training loop
def train(args, model, cam_param, train_dl, test_dl, log_writer):

    # Create the optimisers for the different networks
    optimizer_encoder = torch.optim.Adam(lr=args.encoder_lr, params=model.latent_model.parameters())
    optim_encoder_shed = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=0.9954)

    optimizer_hyper = torch.optim.Adam(lr=args.hyper_lr, params=model.hyper_model.parameters())
    optim_hyper_shed = torch.optim.lr_scheduler.StepLR(optimizer_hyper, step_size=10, gamma=0.9954)

    optimizer_distort = torch.optim.Adam(lr=args.distortion_lr, params=model.distortion_model.parameters())
    optim_distort_shed = torch.optim.lr_scheduler.StepLR(optimizer_distort, step_size=10, gamma=0.9954)

    optimizer_cam = torch.optim.Adam(lr=args.intrinsics_lr, params=cam_param.parameters())
    optim_cam_shed = torch.optim.lr_scheduler.StepLR(optimizer_cam, step_size=20, gamma=0.91)

    # Loss metrics MSE and SSIM
    criterion = torch.nn.MSELoss()
    criterion_ssim = pytorch_ssim.SSIM()

    distort = False

    # Run over epochs
    for epoch in range(args.epochs):
        epoch_time = time.time()

        # Training
        model.train()
        train_photo1_loss_running = 0
        train_photo2_loss_running = 0
        train_ssim_loss_running = 0
        train_encoder_loss_running = 0
        train_position_loss_running = 0
        train_rotation_loss_running = 0

        # Wait until enough learning is done before starting on distortion
        if epoch > args.distortion_start:
            distort = True

        for i, (images, pose, uv, _) in enumerate(train_dl):

            image1 = images[0].to(device)
            image2 = images[1].to(device)
            pose = pose.to(device)
            uv = uv.to(device)
            k = cam_param()
            k = k.to(device)

            # Is labelled data or not
            if i % (1/args.label_ratio):
                labelled = False
            else:
                labelled = True

            # Run forward pass
            predicted_images, pose_est, encoding, distortion_est = model(image1, image2, pose, uv, k,
                                                                         labelled, distort=distort)

            # Calculate Losses
            loss_photo1 = 100*criterion(image1, predicted_images[0])
            loss_photo2 = 100*criterion(image2, predicted_images[1])
            loss_ssim = (2 - criterion_ssim(image1, predicted_images[0]) -
                             criterion_ssim(image2, predicted_images[1])) / 2
            loss_position = criterion(pose[:, :3, 3], pose_est[:, :3, 3])
            loss_rotation = criterion(pose[:, :3, :3], pose_est[:, :3, :3])

            if distortion_est is not None:
                loss_distorion = torch.pow(distortion_est, 2).mean() * 1e-6
            else:
                loss_distorion = 0

            # Small loss on latent vector
            loss_encoder = torch.pow(encoding, 2).mean()*1e-8

            # Change loss depending on if the data is labelled
            if labelled:
                loss = loss_position*20 + loss_rotation*10 + loss_photo1 + loss_photo2 + loss_encoder + loss_distorion
            else:
                loss = loss_photo1 + loss_photo2 + loss_encoder + loss_distorion

            # Ready Optimisers and perform backeard pass
            optimizer_encoder.zero_grad(set_to_none=True)
            optimizer_hyper.zero_grad(set_to_none=True)
            if epoch > args.focal_start:
                optimizer_cam.zero_grad(set_to_none=True)
            if epoch > args.distortion_start:
                optimizer_distort.zero_grad(set_to_none=True)

            loss.backward()

            optimizer_encoder.step()
            optimizer_hyper.step()
            if epoch > args.focal_start:
                optimizer_cam.step()
            if epoch > args.distortion_start:
                optimizer_distort.step()

            # Keep track of losses
            train_photo1_loss_running += loss_photo1.item()
            train_photo2_loss_running += loss_photo2.item()
            train_position_loss_running += loss_position
            train_rotation_loss_running += loss_rotation
            train_ssim_loss_running += loss_ssim.item()
            train_encoder_loss_running += loss_encoder

        # Log training renderings for this epoch
        util.log_predicted(log_writer, epoch, predicted_images[0], predicted_images[1], image1,
                                          images[1].to(device), 'Train - Actual vs Prediction')

        # Evaluate training of validation data
        model.eval()
        test_photo_loss_running = 0
        test_ssim_loss_running = 0
        test_position_loss_running = 0
        test_rotation_loss_running = 0
        
        # Run inference for the test dataset
        for i, (images, pose, uv, _) in enumerate(test_dl):
            image1 = images[0].to(device)
            image2 = images[1].to(device)
            pose = pose.to(device)
            uv = uv.to(device)
            k = cam_param()
            k = k.to(device)

            with torch.no_grad():
                predicted_images, pose_est, encoding, distortion_est = model(image1, image2, pose, uv, k, False, distort=True)

                loss_photo = (criterion(image1, predicted_images[0]) + criterion(image2, predicted_images[1])) * 100

                loss_ssim = (2 - criterion_ssim(image1, predicted_images[0]) -
                        criterion_ssim(image2, predicted_images[1])) / 2
                loss_position = criterion(pose[:, :3, 3], pose_est[:, :3, 3])
                loss_rotation = criterion(pose[:, :3, :3], pose_est[:, :3, :3])

            test_position_loss_running += loss_position.item()
            test_rotation_loss_running += loss_rotation.item()
            test_photo_loss_running += loss_photo.item()
            test_ssim_loss_running += loss_ssim.item()

        # Calculate all losses
        train_photo1_loss_avg = train_photo1_loss_running / (len(train_dl))
        train_photo2_loss_avg = train_photo2_loss_running / (len(train_dl))
        train_encoder_loss_avg = train_encoder_loss_running / (len(train_dl))
        train_ssim_loss_avg = train_ssim_loss_running / (len(train_dl))
        test_ssim_loss_avg = test_ssim_loss_running / (len(test_dl))
        test_photo_loss_avg = test_photo_loss_running / (len(test_dl))
        train_position_loss_avg = train_position_loss_running / (len(train_dl))
        train_rotation_loss_avg = train_rotation_loss_running / (len(train_dl))
        test_rotation_loss_avg = test_rotation_loss_running / (len(test_dl))
        test_position_loss_avg = test_position_loss_running / (len(test_dl))

        # Formate in a dictionary 
        loss_dict = {"Train Photo 1 Loss": train_photo1_loss_avg, "Train Photo 2 Loss": train_photo2_loss_avg,
                     "Test Photo Loss": test_photo_loss_avg, "Train Position Loss": train_position_loss_avg,
                     "Train Rotation Loss": train_rotation_loss_avg,
                     "Test Position Loss": test_position_loss_avg, "Test Rotation Loss": test_rotation_loss_avg,
                     "Train SSIM Loss": train_ssim_loss_avg, "Test SSIM Loss": test_ssim_loss_avg,
                     "Train Encoder Loss": train_encoder_loss_avg, "Learnt Focal Length:": cam_param.f_x}

        # Update optimiser shed
        optim_encoder_shed.step()
        optim_hyper_shed.step()
        if epoch > args.focal_start:
            optim_cam_shed.step()
        if epoch > args.distortion_start:
            optim_distort_shed.step()

        # Log losses and renders
        util.log_run(log_writer, loss_dict, epoch)
        util.log_predicted(log_writer, epoch, predicted_images[0], predicted_images[1], image1, images[1].to(device),
                                          'Eval - Actual vs Prediction')

        # Create save folder if not created
        if not os.path.exists('saved_models/'):
            os.mkdir('saved_models')

        # Save a checkpoint
        if not (epoch+1) % args.checkpoint_period:
            torch.save(model.state_dict(), 'saved_models/{}_checkpoint{}.pt'.format(args.save_path, epoch))
            print('Saving model: {}_checkpoint{}.pt'.format(args.save_path, epoch))

        completed_epoch_time = time.time() - epoch_time

        print('Epoch: {}/{}   Epoch Time: {:.2f}s   Remaining Time: {:.2f}m'
              .format(epoch+1, args.epochs, completed_epoch_time, completed_epoch_time*(args.epochs-epoch)/60))

    # Save final model
    torch.save(model.state_dict(), 'saved_models/{}_final{}.pt'.format(args.save_path, args.epochs))


if __name__ == '__main__':
    main()
