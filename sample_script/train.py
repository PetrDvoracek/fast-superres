# Reference https://github.com/Nhat-Thanh/ESPCN-Pytorch

NAME = "2_japan+in1k+div+flickr+signate"

# %%
batch_size = 64
num_workers = 8
num_epoch = 500
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
        # replace with sigmoid, image is lost
        base = torch.nn.functional.interpolate(X_in, scale_factor=self.scale, mode='bicubic')
        X = base + X
        X_out = clip(X, 0.0, 1.0)
        return X_out


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

        im_orig = PIL.Image.open(p).convert("RGB")

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
        im_small = PIL.Image.open(im_small_p).convert("RGB")

        im_orig_p = self.im_orig_paths[idx]
        im_orig = PIL.Image.open(im_orig_p).convert("RGB")

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


def luma(b):
    b[:, 0, ...] = b[:, 0, ...] * 0.299
    b[:, 1, ...] = b[:, 1, ...] * 0.587
    b[:, 2, ...] = b[:, 2, ...] * 0.114
    return b


def train():
    to_image = transforms.ToPILImage()

    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor(
            (np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        image2 = cv2.cvtColor(
            (np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        return cv2.PSNR(image1, image2)

    device = "cuda"

    model = ESPCN4x()
    model.to(device)
    # model.load_state_dict(torch.load("models/1_in1kval+div+flickr+signate-250.pth"))

    preproc = transforms.ToTensor()
    augmentation = transforms.Compose(
        [
            transforms.RandomCrop(size=512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    # train_dataset, validation_dataset = get_dataset()
    in1k_val = glob.glob("/datasets/imagenet/val/*/*")
    in1k_val_large = [x for x in in1k_val if min(imagesize.get(x)) > 512]
    in1k_train = glob.glob("/datasets/imagenet/train/*/*")
    in1k_train_large = [x for x in in1k_train if min(imagesize.get(x)) > 512]
    train_paths = (
        glob.glob("/datasets/japan_160k/*/*")
        + glob.glob("/datasets/superres/*/*")
        + glob.glob("../train/*")
        + in1k_val_large
        + in1k_train_large
    )
    print(f"n images: {len(train_paths)}")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[
            int(0.3 * num_epoch),
            int(0.5 * num_epoch),
            int(0.65 * num_epoch),
            int(0.8 * num_epoch),
            int(0.9 * num_epoch),
            int(0.95 * num_epoch),
            int(0.97 * num_epoch),
        ],
        gamma=0.7,
    )
    criterion = MSELoss()

    wandb.init(
        project="signate-superres-1374",
        # group=,
        entity="petrdvoracek",
        config=dict(model_id=NAME),
        name=NAME,
    )
    for epoch in trange(num_epoch, desc="EPOCH"):
        should_calc_psnr = (epoch % 10 == 0) or (epoch > epoch - 10)
        try:
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
                if should_calc_psnr:
                    for image1, image2 in zip(output, high_resolution_image):
                        train_psnr += calc_psnr(image1, image2)

                # luma_mse = torch.sum(
                #     (luma(output.detach()) - luma(high_resolution_image.detach())) ** 2
                # )
                # psnr = 20 * torch.log10(1.0 / torch.sqrt(luma_mse))
                # train_psnr.append(psnr)
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
                    if should_calc_psnr:
                        for image1, image2 in zip(output, high_resolution_image):
                            validation_psnr += calc_psnr(image1, image2)
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / len(train_dataset),
                    "val_loss": validation_loss / len(validation_dataset),
                    "examples": wandb.Image(output[:8], caption="asd"),
                }
            )
            if should_calc_psnr:
                wandb.log(
                    {
                        "train_psnr": train_psnr / len(train_dataset),
                        "val_psnr": validation_psnr / len(validation_dataset),
                    }
                )
            if epoch % 10 == 0 or epoch > num_epoch-10:
                torch.save(model.state_dict(), f"models/{NAME}-{epoch}.pth")
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")
            raise

        torch.save(model.state_dict(), f"models/{NAME}.pth")

    torch.save(model.state_dict(), f"models/{NAME}-final.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        f"models/{NAME}.onnx",
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
