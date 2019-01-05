import argparse
import datetime
import functools
import os
import time

import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import multiprocessing, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchtcn
from torchtcn.loss.metric_learning import (LiftedStruct,
                                           TimeWeightedLiftedStruct)
from torchtcn.model.tcn_kl import RNN, create_model, save_model
from torchtcn.utils.comm import get_git_commit_hash,create_dir_if_not_exists
from torchtcn.utils.data import get_data_loader
from torchtcn.utils.dataset import (DoubleViewPairDataset,
                                    MultiViewVideoDataset, ViewPairDataset)
from torchtcn.utils.log import log,set_log_file
from torchtcn.utils.sampler import ViewPairSequenceSampler
from torchtcn.val.alignment import view_pair_alignment_loss
from torchtcn.val.embedding_visualization import visualize_embeddings
from torchvision import transforms
from torchvision.utils import save_image

IMAGE_SIZE = (64, 64)
from jointvae.models import VAE
'''
usage:
# cool implementation: https://github.com/Schlumberger/joint-vae/
pouring:
--load-model ~/tcn_traind/pouring_tcn/model_best.pth.tar \
--load-model ~/tcn_traind/torchtcn/model.pth.tar \
python train_tcn_kl.py \
--train-director ~/tcn_data/multiview-pouring/torch-data-avi/train \
--validation-directory ~/tcn_data/multiview-pouring/torch-data-avi/val

'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--save-folder', type=str,
                        default='~/tcn_traind_kl/torchtcn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--train-directory', type=str, default='~/data/train/')
    parser.add_argument('--validation-directory',
                        type=str, default='~/tcn_data/val')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr-start', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--beta', type=float, default=1.)
    return parser.parse_args()


def get_dataloader_train(dir_vids, num_views, batch_size, ):
    transformer_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE[0]),  # TODO reize here Resize()
        # transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               # hue=0.1,
                               saturation=0.3),
        transforms.ToTensor(),
        # normalize
    ])

    sampler = None
    shuffle = True
    # only one view pair in batch
    # sim_frames = 5
    transformed_dataset_train = DoubleViewPairDataset(vid_dir=dir_vids,
                                                      number_views=num_views,
                                                      # std_similar_frame_margin_distribution=sim_frames,
                                                      transform_frames=transformer_train)
    # sample so that only one view pairs is in a batch
#     sampler = ViewPairSequenceSampler(dataset=transformed_dataset_train,
                                      # examples_per_sequence=batch_size,
                                      # batch_size=batch_size)

    dataloader_train = DataLoader(transformed_dataset_train, drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  num_workers=4,
                                  pin_memory=use_cuda)

    return dataloader_train





if __name__ == '__main__':

    args = get_args()
    log.setLevel(logging.INFO)
    set_log_file(os.path.join(args.save_folder, "train.log"))

    log.info("torchtcn commit hash: {}".format(get_git_commit_hash(torchtcn.__file__)))
    writer = SummaryWriter(os.path.join(
        os.path.expanduser(args.save_folder), "tensorboard_log"))
    create_dir_if_not_exists(os.path.join(args.save_folder, "images"))
    use_cuda = torch.cuda.is_available()
    # LOSS

    if use_cuda:
        torch.cuda.seed()

    from jointvae.models import VAE
    from utils.dataloaders import get_mnist_dataloaders
    # dataloader_train, test_loader = get_mnist_dataloaders(batch_size=64)
    dataloader_train = get_dataloader_train(args.train_directory, args.num_views,
                                            args.batch_size, )
# Latent distribution will be joint distribution of 10 gaussian normal distributions
# and one 10 dimensional Gumbel Softmax distribution
    latent_spec = {'cont': 32,
                'disc': [10]}

    model = VAE(latent_spec=latent_spec, img_size=(3, IMAGE_SIZE[0], IMAGE_SIZE[0]),use_cuda=use_cuda)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    from jointvae.training import Trainer

    # Define the capacities
    # Continuous channels
    cont_capacity = [0.0, 5.0, 25000, 30.0]  # Starting at a capacity of 0.0, increase this to 5.0
                                             # over 25000 iterations with a gamma of 30.0
    # Discrete channels
    disc_capacity = [0.0, 5.0, 25000, 30.0]  # Starting at a capacity of 0.0, increase this to 5.0
                                             # over 25000 iterations with a gamma of 30.0

    # Build a trainer
    trainer = Trainer(model, optimizer,
                      cont_capacity=cont_capacity,
                      disc_capacity=disc_capacity,use_cuda=use_cuda)



    # Build a visualizer which will be passed to trainer to visualize progress during training
    from viz.visualize import Visualizer

    viz = Visualizer(model)



    # Train model for 10 epochs
    # Note this should really be a 100 epochs and trained on a GPU, but this is just to demo

    trainer.train(dataloader_train, epochs=100, save_training_gif=('./training.gif', viz))


