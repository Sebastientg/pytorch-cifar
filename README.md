# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage

### Arguments
- `--optimizer`: `SGD` (default) or `Adam`
- `--lr`: learning rate (default: `0.1`)
- `--epochs`: number of epochs (default: `10`)
- `--model1`: required for real training
- `--model2`: optional, enables two-model comparison
- `--resume`: resume from the last saved checkpoint for each model
- `--mock`: mock mode with randomized metrics

### Cases
- **Case 0 (mock):** 2 curves (train/test)
- **Case 1 (`--model1` only):** 2 curves (train/test)
- **Case 2 (`--model1` + `--model2`):** 4 curves (train/test for each model)

Graph titles include optimizer and learning rate. Graphs are saved in the `graphs` directory.

### Examples
```bash
# Case 0: mock mode
python main.py --mock --epochs 10 --optimizer SGD --lr 0.1

# Case 1: single model
python main.py --model1 ResNet18 --epochs 10 --optimizer Adam --lr 0.001

# Case 2: two-model comparison
python main.py --model1 ResNet18 --model2 VGG19 --epochs 10 --optimizer SGD --lr 0.1

# Resume an interrupted run
python main.py --model1 ResNet18 --epochs 10 --optimizer SGD --lr 0.1 --resume
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

## Graph Generation

The script plots epochs on the x-axis and accuracy on the y-axis:
- single-model runs: 2 curves
- two-model runs: 4 curves

