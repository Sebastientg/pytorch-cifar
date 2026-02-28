'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import matplotlib.pyplot as plt

# Create directory for graphs if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Initialize lists to store metrics
epochs = []
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--optimizer', default='SGD',
                    choices=['SGD', 'Adam'], help='Optimizer to use')
parser.add_argument('--model', default='SimpleDLA', choices=['VGG19', 'ResNet18', 'PreActResNet18', 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d',
                    'MobileNet', 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA'], help='Model to use')
parser.add_argument('--mock', action='store_true',
                    help='Enable mock mode for quick testing')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model_dict = {
    'VGG19': lambda: VGG('VGG19'),
    'ResNet18': ResNet18,
    'PreActResNet18': PreActResNet18,
    'GoogLeNet': GoogLeNet,
    'DenseNet121': DenseNet121,
    'ResNeXt29_2x64d': ResNeXt29_2x64d,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DPN92': DPN92,
    'ShuffleNetG2': ShuffleNetG2,
    'SENet18': SENet18,
    'ShuffleNetV2': lambda: ShuffleNetV2(1),
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF,
    'SimpleDLA': SimpleDLA
}

net = model_dict[args.model]()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Dynamically select the optimizer
optimizer_dict = {
    'SGD': lambda: optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
    'Adam': lambda: optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
}

optimizer = optimizer_dict[args.optimizer]()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        train_losses.append(train_loss / (batch_idx + 1))
        train_accuracies.append(100. * correct / total)


# Testing
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            test_losses.append(test_loss / (batch_idx + 1))
            test_accuracies.append(100. * correct / total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == '__main__':
    if args.mock:
        print('==> Running in mock mode..')
        import random

        # Simulate training and testing with dummy data
        for epoch in range(1, 6):  # Use a small number of epochs for testing
            epochs.append(epoch)

            # Generate random values for losses and accuracies
            train_loss = random.uniform(0.5, 1.5)
            train_acc = random.uniform(70, 90)
            test_loss = random.uniform(0.5, 1.5)
            test_acc = random.uniform(70, 90)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.3f}, Test Acc={test_acc:.2f}%")

    else:
        for epoch in range(start_epoch, start_epoch + 10):  # Example: Run for 10 epochs
            epochs.append(epoch)
            train(epoch)
            test(epoch)
            scheduler.step()

            # Append metrics for graph generation
            train_losses.append(train_loss / len(trainloader))
            train_accuracies.append(100. * correct / total)
            test_losses.append(test_loss / len(testloader))
            test_accuracies.append(100. * correct / total)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Testing Accuracy', color='orange')
    plt.title(
        f'Training and Testing Curves for {args.model} with {args.optimizer}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    # Ensure epochs are whole numbers
    plt.xticks(range(min(epochs), max(epochs) + 1))
    # Ensure accuracy has decimals
    plt.yticks([round(i, 1) for i in plt.yticks()[0]])
    plt.legend()
    plt.figtext(0.5, 0.01, 'Generated by PyTorch CIFAR10 Training Script',
                wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(f'graphs/{args.model}_{args.optimizer}_training_curve.png')
    plt.close()

    print('Graph generated and saved in the graphs directory.')
