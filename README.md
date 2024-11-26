# Image Segmenter (Mask R-CNN)

This code enables the creation of a model that segments objects in a images based on a specified data structure.

## 🛠️ Config

To execute the code set the following variables in the fit.yaml file.

- `trainer:logge:save_dir`: Output path for saving model weights.
- `model:num_classes`: Total number of classes.
- `model:num_channels`: Total number of image channels (default 3).
- `model:learning_rate`: Learning rate for the optimizer.
- `model:data_path`: Path to the data, following the structure outlined in the Data Structure section.
- `model:input_size`: Shape used for the input image, e.g., 256 for 256x256.
- `model:batch_size`: Batch size used during the training process.
- `model:workers`: Number of CPU cores allocated for the data loader.

## 🗂️ Data structure

The data structure consists of two main directories: `train` and `validation`. Each of these directories contains two subdirectories named `images` and `masks`. The `images` directory contains multichannel images or RGB images, while the `masks` directory contains the corresponding image labels in 8-bit image format. Both the `images` and `masks` directories contain NumPy files (`.npy`), named with consecutive numbers, e.g., `0.npy`, `1.npy`, `n.npy`. The relationship between images and masks is one-to-one, meaning that for each image file, there is a corresponding mask file with the same consecutive number, e.g., `{images/0.npy, masks/0.npy}`.


```
dataset
├── train
│   ├── images
│   ├────── 0.npy
│   ├────── 1.npy
│   ├────── ...
│   ├────── n.npy
│   ├── masks
│   ├────── 0.npy
│   ├────── 1.npy
│   ├────── ...
│   └────── n.npy
├── validation
│   ├── images
│   ├────── 0.npy
│   ├────── 1.npy
│   ├────── ...
│   ├────── n.npy
│   ├── masks
│   ├────── 0.npy
│   ├────── 1.npy
│   ├────── ...
│   └────── n.npy
```

## ⚙️ Usage

To execute training process run the following in console/terminal:

```bash
python main.py fit --config fit.yaml
```