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

# Initialize metrics per model key
metrics = {
    'model1': {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    },
    'model2': {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
}

# CLI arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Global training settings
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--optimizer', default='SGD',
                    choices=['SGD', 'Adam'], help='Optimizer to use')
# Model selection (model2 is optional for comparison mode)
parser.add_argument('--model1', choices=['VGG19', 'ResNet18', 'PreActResNet18', 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d',
                    'MobileNet', 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA'], help='First model to use')
parser.add_argument('--model2', choices=['VGG19', 'ResNet18', 'PreActResNet18', 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d',
                    'MobileNet', 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA'], help='Second model to use (optional)')
# Epoch count and optional mock mode
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--mock', action='store_true',
                    help='Enable mock mode for quick testing')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Resume state is tracked per model key
resume_state = {
    'model1': {'start_epoch': 0, 'best_acc': 0.0},
    'model2': {'start_epoch': 0, 'best_acc': 0.0},
}


def build_optimizer(model):
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)


def checkpoint_path(metric_key):
    return f'./checkpoint/{metric_key}_ckpt.pth'


def load_checkpoint_if_needed(model, optimizer, metric_key):
    # Load model/optimizer/epoch only when --resume is set
    if not args.resume:
        return
    ckpt_path = checkpoint_path(metric_key)
    if not os.path.isfile(ckpt_path):
        print(f'==> No checkpoint found for {metric_key} at {ckpt_path}')
        return
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    resume_state[metric_key]['start_epoch'] = checkpoint['epoch'] + 1
    resume_state[metric_key]['best_acc'] = checkpoint.get(
        'best_acc', checkpoint.get('acc', 0.0))
    print(
        f"==> Resumed {metric_key} from epoch {resume_state[metric_key]['start_epoch']}")


def save_checkpoint(model, optimizer, metric_key, epoch, acc):
    # Save latest training state so Ctrl+C can resume later
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    resume_state[metric_key]['best_acc'] = max(
        resume_state[metric_key]['best_acc'], acc)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'best_acc': resume_state[metric_key]['best_acc'],
    }
    torch.save(state, checkpoint_path(metric_key))


# Training
def train(epoch, model, optimizer, metric_key):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    metrics[metric_key]['train_loss'].append(train_loss / len(trainloader))
    metrics[metric_key]['train_acc'].append(100. * correct / total)


# Testing
def test(epoch, model, metric_key):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    metrics[metric_key]['test_loss'].append(test_loss / len(testloader))
    metrics[metric_key]['test_acc'].append(100. * correct / total)
    return 100. * correct / total


def plot_single_model(model_name, epochs):
    x = list(range(1, len(metrics['model1']['train_acc']) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x, metrics['model1']['train_acc'],
             label=f'{model_name} Train', color='blue')
    plt.plot(x, metrics['model1']['test_acc'],
             label=f'{model_name} Test', color='orange')
    plt.title(f'{model_name} | Optimizer: {args.optimizer}, LR: {args.lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.xlim(1, 20)
    plt.xticks(range(1, 21))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/{model_name}_{args.optimizer}_lr{args.lr}_curves.png')
    plt.close()


def plot_two_models(model1_name, model2_name, epochs):
    x_model1 = list(range(1, len(metrics['model1']['train_acc']) + 1))
    x_model2 = list(range(1, len(metrics['model2']['train_acc']) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x_model1, metrics['model1']['train_acc'],
             label=f'{model1_name} Train', linestyle='--')
    plt.plot(x_model1, metrics['model1']['test_acc'],
             label=f'{model1_name} Test', linestyle='-')
    plt.plot(x_model2, metrics['model2']['train_acc'],
             label=f'{model2_name} Train', linestyle='--')
    plt.plot(x_model2, metrics['model2']['test_acc'],
             label=f'{model2_name} Test', linestyle='-')
    plt.title(
        f'{model1_name} vs {model2_name} | Optimizer: {args.optimizer}, LR: {args.lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.xlim(1, 20)
    plt.xticks(range(1, 21))
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f'graphs/{model1_name}_vs_{model2_name}_{args.optimizer}_lr{args.lr}_curves.png')
    plt.close()


if __name__ == '__main__':
    # Case 0: mock mode (randomized metrics)
    if args.mock:
        print('==> Running in mock mode..')
        import random

        # Simulate training and testing with dummy data
        for epoch in range(1, args.epochs + 1):

            # Generate random values for losses and accuracies
            train_loss = random.uniform(0.5, 1.5)
            train_acc = random.uniform(70, 90)
            test_loss = random.uniform(0.5, 1.5)
            test_acc = random.uniform(70, 90)

            metrics['model1']['train_loss'].append(train_loss)
            metrics['model1']['train_acc'].append(train_acc)
            metrics['model1']['test_loss'].append(test_loss)
            metrics['model1']['test_acc'].append(test_acc)

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.3f}, Test Acc={test_acc:.2f}%")
        plot_single_model('MockModel', args.epochs)

    else:
        # Case 1 / 2: real training (one model or two-model comparison)
        if not args.model1:
            raise ValueError('Please provide --model1, or use --mock.')

        model1 = model_dict[args.model1]().to(device)
        if device == 'cuda':
            model1 = torch.nn.DataParallel(model1)
        optimizer1 = build_optimizer(model1)
        load_checkpoint_if_needed(model1, optimizer1, 'model1')

        for epoch in range(resume_state['model1']['start_epoch'], args.epochs):
            train(epoch, model1, optimizer1, 'model1')
            acc1 = test(epoch, model1, 'model1')
            save_checkpoint(model1, optimizer1, 'model1', epoch, acc1)

        # Case 2: two models -> 4 curves
        if args.model2:
            model2 = model_dict[args.model2]().to(device)
            if device == 'cuda':
                model2 = torch.nn.DataParallel(model2)
            optimizer2 = build_optimizer(model2)
            load_checkpoint_if_needed(model2, optimizer2, 'model2')

            for epoch in range(resume_state['model2']['start_epoch'], args.epochs):
                train(epoch, model2, optimizer2, 'model2')
                acc2 = test(epoch, model2, 'model2')
                save_checkpoint(model2, optimizer2, 'model2', epoch, acc2)

            plot_two_models(args.model1, args.model2, args.epochs)
        else:
            # Case 1: one model -> 2 curves
            plot_single_model(args.model1, args.epochs)

    print('Graph generated and saved in the graphs directory.')
