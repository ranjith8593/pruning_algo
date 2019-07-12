import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from core.model.lenet import LeNet
from util.accuracy_holder import AccuracyHolder
from util.model_train import train, top1_accuracy


def get_mnist_dataloaders(root, batch_size):
    """
    This function should return a pair of torch.utils.data.DataLoader.
    The first element is the training loader, the second is the test loader.
    Those loaders should have been created from the MNIST dataset available in torchvision.

    For the training set, please preprocess the images in this way:
        - Resize to 32x32
        - Randomly do a horizontal flip
        - Normalize using mean 0.1307 and std 0.3081

    For the training set, please preprocess the images in this way:
        - Resize to 32x32
        - Normalize using mean 0.1307 and std 0.3081

    :param root: Folder where the dataset will be downloaded into.
    :param batch_size: Number of samples in each mini-batch
    :return: tuple
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dt = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    train_dt_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    test_dt = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    test_dt_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=True)
    return (train_dt_loader, test_dt_loader)


def get_accuracy_top1(model, data_loader):
    """
    This function should return the top1 accuracy% of the model on the given data loader.
    :param model: LeNet object
    :param data_loader: torch.utils.data.DataLoader
    :return: float
    """
    top1 = AccuracyHolder('Acc')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            output = model(images)
            acc1 = top1_accuracy(output, target)
            top1.update(acc1[0], images.size(0))
    return top1.avg.cpu().numpy()[0]


def prune_model(model, threshold):
    """
    This function should set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    :param model: LeNet object
    :param threshold: float
    """
    modified_weights = []
    for param in model.parameters():
        if len(param.data.size()) != 1:
            pruned = param.data.abs() <= threshold
            modified_weights.append(pruned.float())
    return modified_weights


def prune_model_finetune(model, train_loader, test_loader, threshold):
    """
    This function should first set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    Then, it should finetune the model by making sure that the weights that were set to zero remain zero after the gradient descent steps.
    :param model: LeNet object
    :param train_loader: training set torch.utils.data.DataLoader
    :param test_loader: testing set torch.utils.data.DataLoader
    :param threshold: float
    """
    mask = prune_model(model, threshold)
    net = LeNet()
    net.load_state_dict(model.state_dict())
    net.set_mask(mask)
    # Retraining
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)
    for e in range(1, 8):
        train(e, train_loader, net, criterion, optimizer)
    print("top1 accuracy")
    print(get_accuracy_top1(net, test_loader))


if __name__ == '__main__':
    model = torch.load('lenet.pth')
    train_loader, test_loader = get_mnist_dataloaders("./data/mnist", 256)
    weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(weights), 50)
    print(threshold)
    prune_model_finetune(model, train_loader, test_loader, threshold)
