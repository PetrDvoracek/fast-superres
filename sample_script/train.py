# Sample scripts for training and model generation
# Same contents as notebook(train.ipynb)
# Please refer to the README to set up the environment in advance and place the dataset you have obtained in the following format
# train.py
# dataset/ # dataset you are distributing
# + train/
# + validation/
# + 0.25x/
# + original/


# training parameters

NAME = "0_sample_defaultinit"

# %%
batch_size = 64
num_workers = 16
num_epoch = 100
learning_rate = 1e-3

import os
import random
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import timm
import torch
from torch import nn, clip, tensor, Tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange
import onnxruntime as ort
import numpy as np
import datetime
from typing import Tuple
import PIL
from PIL.Image import Image
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
from tqdm import tqdm
import wandb
import imagesize


# Structure definition of the 4x magnified sample model (ESPCN)
# Reference https://github.com/Nhat-Thanh/ESPCN-Pytorch
# The inputs to the model are assumed to be 4-dimensional inputs in N, C, H, W, with channels in the order R, G, B and pixel values normalized to 0~1.
# The output will also be in the same format, with 4x vertical and horizontal resolution (H, W).


class ESPCN4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=5, padding=2
        )
        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.conv_4 = nn.Conv2d(
            in_channels=32,
            out_channels=(1 * self.scale * self.scale),
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X_in: tensor) -> tensor:
        X = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X = self.act(self.conv_1(X))
        X = self.act(self.conv_2(X))
        X = self.act(self.conv_3(X))
        X = self.conv_4(X)
        X = self.pixel_shuffle(X)
        X = X.reshape(-1, 3, X.shape[-2], X.shape[-1])
        X_out = clip(X, 0.0, 1.0)
        return X_out


# %%

# Data set definition
# This class reads the provided training and evaluation image sets (high resolution + low resolution).
# The training image is a 512px square cut from the original image and is used as the correct image. The correct image is also used as the input image (TrainDataSet), which is a 1/4 size smaller than the correct image.
# Since the evaluation images are provided in sets of high and low resolution, the low resolution image is used as the input image and the high resolution image is used as the correct image (ValidationDataSet).


class DatasetGeneral(torch.utils.data.Dataset):
    def __init__(
        self,
        im_paths,
        transform,
        n_im_per_epoch=2000,
        aug=None,
    ):
        self.im_paths = im_paths
        self.transform = transform
        self.aug = aug
        self.len = n_im_per_epoch

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        idx = random.randint(0, len(self.im_paths) - 1)
        p = self.im_paths[idx]

        im_orig = PIL.Image.open(p)

        if self.aug is not None:
            im_orig = self.aug(im_orig)

        im_small = transforms.Resize(
            (im_orig.size[0] // 4, im_orig.size[1] // 4),
            transforms.InterpolationMode.BICUBIC,
        )(im_orig.copy())

        im_orig = transforms.ToTensor()(im_orig)
        im_small = transforms.ToTensor()(im_small)

        return im_small, im_orig


class DatasetValSignate(torch.utils.data.Dataset):
    def __init__(
        self, images_small_Wild, transform, small_id="0.25x", orig_id="original"
    ):
        self.im_small_paths = glob.glob(images_small_Wild)
        self.im_orig_paths = [x.replace(small_id, orig_id) for x in self.im_small_paths]
        self.transform = transform

    def __len__(self):
        return len(self.im_small_paths)

    def __getitem__(self, idx):
        im_small_p = self.im_small_paths[idx]
        im_small = PIL.Image.open(im_small_p)

        im_orig_p = self.im_orig_paths[idx]
        im_orig = PIL.Image.open(im_orig_p)

        return (
            self.transform(im_small),
            self.transform(im_orig),
        )


def seedme(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


# Learning
# Train the defined model with pytorch.
# Adjust batch size and other parameters according to the VRAM of your GPU.
# The logs of the training are saved in the log folder.
# After learning, torch.onnx.export is called to convert the model to an ONNX model.
# In this case, set dynamic_axes so that opset=17, the model input name is input, the model output name is output, and the model input shape is (1, 3, height, width).
# (In this example, after setting dummy input of (1, 3, 128, 128), set dynamic_axes to shape[2] and shape[3] so that the input shape of the model is (1, 3, height, width).
def train():
    to_image = transforms.ToPILImage()

    wandb.init(
        project="signate-superres-1374",
        # group=,
        entity="petrdvoracek",
        config=dict(model_id=NAME),
        name=NAME,
    )

    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor(
            (np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        image2 = cv2.cvtColor(
            (np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        return cv2.PSNR(image1, image2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESPCN4x()
    model.to(device)
    writer = SummaryWriter("log")

    # transform_for_net = A.Compose([A.ToFloat(), ToTensorV2()])
    # augmentation = A.Compose(
    #     [
    #         # A.PadIfNeeded(512, 512, always_apply=True),
    #         A.RandomCrop(512, 512),
    #         A.HorizontalFlip(),
    #         A.VerticalFlip(),
    #     ]
    # )
    preproc = transforms.ToTensor()
    augmentation = transforms.Compose(
        [
            transforms.RandomCrop(size=512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    # train_dataset, validation_dataset = get_dataset()
    in1k = glob.glob("/datasets/imagenet/val/*/*")
    in1k_large = [x for x in in1k if min(imagesize.get(x)) > 512]
    train_paths = (
        glob.glob("/datasets/superres/*/*") + glob.glob("../train/*") + in1k_large
    )
    train_dataset = DatasetGeneral(
        train_paths,
        transform=preproc,
        aug=augmentation,
        n_im_per_epoch=850 * 10,
    )
    validation_dataset = DatasetValSignate("../validation/0.25x/*", transform=preproc)
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7)
    criterion = MSELoss()

    for epoch in trange(num_epoch, desc="EPOCH"):
        try:
            # 学習
            model.train()
            train_loss = 0.0
            validation_loss = 0.0
            train_psnr = 0.0
            validation_psnr = 0.0
            for idx, (low_resolution_image, high_resolution_image) in tqdm(
                enumerate(train_data_loader),
                desc=f"EPOCH[{epoch}] TRAIN",
                total=len(train_data_loader),
            ):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                for idx, (low_resolution_image, high_resolution_image) in tqdm(
                    enumerate(validation_data_loader),
                    desc=f"EPOCH[{epoch}] VALIDATION",
                    total=len(validation_data_loader),
                ):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):
                        validation_psnr += calc_psnr(image1, image2)
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / len(train_dataset),
                    "train_psnr": train_psnr / len(train_dataset),
                    "val_loss": validation_loss / len(validation_dataset),
                    "val_psnr": validation_psnr / len(validation_dataset),
                    "examples": wandb.Image(output[:8], caption="asd"),
                }
            )
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()

    torch.save(model.state_dict(), "model.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )


# Inference with ONNX models (equivalent to running on SIGNATE)
# Check the model trained and transformed by pytorch by inference with onnxruntime.
# Images of the inference results are generated in the output folder.
# We will also measure the processing time in the environment at hand, although it is simple.
def inference_onnxruntime():
    input_image_dir = Path("dataset/validation/0.25x")
    output_image_dir = Path("output")
    output_image_dir.mkdir(exist_ok=True, parents=True)

    sess = ort.InferenceSession(
        "model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_images = []
    output_images = []
    output_paths = []

    print("load image")
    for image_path in input_image_dir.iterdir():
        output_iamge_path = output_image_dir / image_path.relative_to(input_image_dir)
        input_image = cv2.imread(str(image_path))
        input_image = (
            np.array(
                [cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))],
                dtype=np.float32,
            )
            / 255
        )
        input_images.append(input_image)
        output_paths.append(output_iamge_path)

    print("inference")
    start_time = datetime.datetime.now()
    for input_image in input_images:
        output_images.append(sess.run(["output"], {"input": input_image})[0])
    end_time = datetime.datetime.now()

    print("save image")
    for output_path, output_image in zip(output_paths, output_images):
        output_image = cv2.cvtColor(
            (output_image.transpose((0, 2, 3, 1))[0] * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(output_path), output_image)

    print(
        f"inference time: {(end_time - start_time).total_seconds() / len(input_images)}[s/image]"
    )


# PSNR calculation (with comparison to conventional methods)
# PSNR measurement is performed on images resulting from inference with onnxruntime.
# This script also compares with conventional methods.
def calc_and_print_PSNR():
    input_image_dir = Path("dataset/validation/0.25x")
    output_image_dir = Path("output")
    original_image_dir = Path("dataset/validation/original")
    output_label = ["MODEL", "NEAREST", "BILINEAR", "BICUBIC"]
    output_psnr = [0.0, 0.0, 0.0, 0.0]
    original_image_paths = list(original_image_dir.iterdir())
    for image_path in tqdm(original_image_paths):
        input_image_path = input_image_dir / image_path.relative_to(original_image_dir)
        output_iamge_path = output_image_dir / image_path.relative_to(
            original_image_dir
        )
        input_image = cv2.imread(str(input_image_path))
        original_image = cv2.imread(str(image_path))
        espcn_image = cv2.imread(str(output_iamge_path))
        output_psnr[0] += cv2.PSNR(original_image, espcn_image)
        h, w = original_image.shape[:2]
        output_psnr[1] += cv2.PSNR(
            original_image,
            cv2.resize(input_image, (w, h), interpolation=cv2.INTER_NEAREST),
        )
        output_psnr[2] += cv2.PSNR(
            original_image,
            cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LINEAR),
        )
        output_psnr[3] += cv2.PSNR(
            original_image,
            cv2.resize(input_image, (w, h), interpolation=cv2.INTER_CUBIC),
        )
    for label, psnr in zip(output_label, output_psnr):
        print(f"{label}: {psnr / len(original_image_paths)}")


if __name__ == "__main__":
    seedme(42)
    train()
    inference_onnxruntime()
    calc_and_print_PSNR()
