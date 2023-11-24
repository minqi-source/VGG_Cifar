#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from options import args_parser
from utils import get_dataset, test_inference
from models import CNNMnist, CNNCifar, VGGCifar, VGGMnist


if __name__ == '__main__':
    args = args_parser()
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            model = CNNCifar(args=args)
    elif args.model == 'vgg':
        # VGG
        if args.dataset == 'cifar':
            model = VGGCifar(args=args)
        elif args.dataset == 'mnist':
            model = VGGMnist(args=args)            
        else:
            exit('Error: unrecognized model or dataset!!!')
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    model.to(device)
    model.train()
    print(model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=args.BS, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, 
                    batch_idx * len(images), 
                    len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), 
                    loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # testing
    test_acc, test_loss = test_inference(args, model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
