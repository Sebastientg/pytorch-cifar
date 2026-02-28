# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Updated Training Instructions

You can now specify the optimizer and model directly from the command line using the following arguments:

- `--optimizer`: Choose the optimizer to use. Options are `SGD` (default) and `Adam`.
- `--model`: Choose the model to train. Available options include:
  - `VGG19`
  - `ResNet18`
  - `PreActResNet18`
  - `GoogLeNet`
  - `DenseNet121`
  - `ResNeXt29_2x64d`
  - `MobileNet`
  - `MobileNetV2`
  - `DPN92`
  - `ShuffleNetG2`
  - `SENet18`
  - `ShuffleNetV2`
  - `EfficientNetB0`
  - `RegNetX_200MF`
  - `SimpleDLA` (default)

### Examples

Start training with the default optimizer (SGD) and model (SimpleDLA):
```
python main.py
```

Specify a different optimizer and model:
```
python main.py --optimizer Adam --model ResNet18
```

Resume training with a specific learning rate:
```
python main.py --resume --lr=0.01
```

## Mock Mode

For quick testing without running the full training process, you can use the `--mock` argument. This will simulate training and testing with dummy data and generate a graph.

### Example
Run in mock mode:
```
python main.py --mock
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

During training and testing, the script generates a graph showing the training and testing accuracy over epochs. The graph is saved in the `graphs` directory with a filename based on the selected model and optimizer.

### Example Output
Graph for `SimpleDLA` with `SGD`:
```
graphs/SimpleDLA_SGD_training_curve.png
```

