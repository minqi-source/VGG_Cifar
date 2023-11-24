#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

def get_dataset(args):
    # Returns train and test datasets
    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'mnist':
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    return train_dataset, test_dataset

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    LearningRate : {args.lr}')
    print(f'    DataSet  : {args.dataset}')
    print(f'    Rounds   : {args.rounds}\n')
    print(f'    Batch size   : {args.BS}')
    print(f'    Epochs       : {args.E}\n')
    return


def test_inference(args, model, test_dataset):
    # Returns the test accuracy and loss.
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=args.BS,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss