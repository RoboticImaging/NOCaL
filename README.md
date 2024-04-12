# NOCaL: Calibration-Free Semi-Supervised Learning of Odometry and Camera Intrinsics

We introduce a semi-supervised learning framework capable of interpreting previously unseen cameras without calibration. We present this in our paper:


[NOCaL: Calibration-Free Semi-Supervised Learning of Odometry and Camera Intrinsics](https://arxiv.org/abs/2210.07435)

<div align="center">
<img src="https://ryanbgriffiths.github.io/images/publications/NOCaL/nvs.gif">
 </div>

Authors: [Ryan Griffiths](https://ryanbgriffiths.github.io), [Jack Naylor](https://nackjaylor.github.io), [Donald G. Dansereau](https://roboticimaging.org/)

Project Page: [roboticimaging.org/Projects/NOCaL/](https://roboticimaging.org/Projects/NOCaL/) (gallery, results etc.)

If you have any questions or issues feel free to reach out.

# Setup


Get the code and create the necessary conda environment using the following:
    
    git clone https://github.com/RoboticImaging/NOCaL.git
    cd NOCaL
    conda env create -f environment.yml

<details>
  <summary> For Detailed Dependencies </summary>

  ## Dependencies
  - python=3.7
  - cudatoolkit=11.0.221
  - pytorch=1.7.1
  - torchvision=0.8.2
  - numpy=1.20.1
  - opencv=3.4.2
  - pandas=1.2.4
  - scipy=1.6.2
  - configargparse
  - scikit-image=0.18.1
  - tensorboardx=2.2
  - matplotlib=3.3.4
  - natsort=7.1.1
  - ordered-set=4.0.2
  - tensorboard
</details>

# Running

Update the config file (data_path) to link to where the dataset is stored, if you want to try the dataset used in this work you can find it [here](http://mediathinktank.com/datarequest/).

Run the following command to start the training process:

    conda activate nocal
    python train.py --config LFOdo_config.txt

Logging is done through tensorboard, you can see the progress/results by running this and following the link provided:

    tensorboard --logdir tensorboard_data


# Using your own data
If you are using your own data and want to use the existing dataloader please formate your data as follows:

    ├── data
    │   ├── scene0
    │   │   ├── cam0
    |   |   |   └── data
    |   |   |   |   └── image1.png
    |   |   |   |   └── image2.png
    |   |   |   |   └── ...
    |   |   |   └── data.csv (name, filename)
    │   │   └── cameraInfo.txt (k matrix)                
    │   │   └── poses.gt (image,x,y,z,q_w,q_x,q_y,q_z)
    |   ├── scene...

You can then use the existing dataloader, by changing the dataset in the config file. Otherwise you can create a new dataloader for your dataset.

Create a new config file and use this file when running the commands above.

See _LFOdo_config.txt_ for configuration options.

# Citation

If you find our work useful, please cite the below paper:  

    @article{griffiths2022nocal,  
      title = {{NOCaL}: Calibration-Free Semi-Supervised Learning of Odometry and Camera Intrinsics},  
      author = {Ryan Griffiths and Jack Naylor and Donald G. Dansereau},  
      journal = {arXiv preprint arXiv:2210.07435},  
      year = {2022}  
    }
