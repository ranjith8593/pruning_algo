import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from core.model.lenet import LeNet

"""
    Deeplite coding challenge.
	The main goal of this coding challenge is to implement a very simple pruning algorithm for lenet. There are two steps 
	to implement this coding challenge. 
	Step 1:
		Implement the pruning algorithm to remove weights which are smaller than the given threshold (prune_model)
	Step 2:
		As you may know after pruning, the accuracy drops a lot. To recover the accuracy, we need to do the fine-tuning.
		It means, we need to retrain the network for few epochs. Use prune_model method which you have implemented in 
		step 1 and then fine-tune the network to regain the accuracy drop (prune_model_finetune)
    
    *** The pretrained lenet has been provided (lenet.pth)
    *** You need to install torch 0.3.1 on ubuntu
    *** You can use GPU or CPU for fine-tuning
"""


class AccuracyHolder(object):
    """Computes and stores the average and current value"""
    avg = None
    val = 0
    sum = 0
    count = 0

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, data_train_loader, net, criterion, optimizer):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    cur_batch_win = None
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)
        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()
    torch.save(net, "lenet_train_mod.pth")


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
    train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    test_dt = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    return (train_loader, test_loader)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    acc = None
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            output = model(images)
            loss = criterion(output, target)
            acc1 = accuracy(output, target)
            acc = top1.update(acc1[0], images.size(0))
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
    model = torch.load('lenet_train.pth')
    train_loader, test_loader = get_mnist_dataloaders("./data/mnist", 256)
    weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(weights), 50)
    print(threshold)
    prune_model_finetune(model, train_loader, test_loader, threshold)
