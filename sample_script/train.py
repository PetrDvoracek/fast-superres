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

# %%
batch_size = 64
num_workers = 16
num_epoch = 100
learning_rate = 1e-3

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
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        self.conv_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(
            in_channels=32,
            out_channels=(1 * self.scale * self.scale),
            kernel_size=3,
            padding=1,
        )
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

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


class DataSetBase(data.Dataset, ABC):
    def __init__(self, image_path: Path):
        self.images = list(image_path.iterdir())
        self.max_num_sample = len(self.images)

    def __len__(self) -> int:
        return self.max_num_sample

    @abstractmethod
    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        pass

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image_path = self.images[index % len(self.images)]
        high_resolution_image = self.preprocess_high_resolution_image(
            PIL.Image.open(image_path)
        )
        low_resolution_image = self.get_low_resolution_image(
            high_resolution_image, image_path
        )
        return transforms.ToTensor()(low_resolution_image), transforms.ToTensor()(
            high_resolution_image
        )


class TrainDataSet(DataSetBase):
    def __init__(self, image_path: Path, num_image_per_epoch: int = 2000):
        super().__init__(image_path)
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return transforms.Resize(
            (image.size[0] // 4, image.size[1] // 4),
            transforms.InterpolationMode.BICUBIC,
        )(image.copy())

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose(
            [
                transforms.RandomCrop(size=512),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )(image)


class ValidationDataSet(DataSetBase):
    def __init__(
        self, high_resolution_image_path: Path, low_resolution_image_path: Path
    ):
        super().__init__(high_resolution_image_path)
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return PIL.Image.open(
            self.low_resolution_image_path
            / path.relative_to(self.high_resolution_image_path)
        )


def get_dataset() -> Tuple[TrainDataSet, ValidationDataSet]:
    return TrainDataSet(Path("./dataset/train"), 850 * 10), ValidationDataSet(
        Path("./dataset/validation/original"), Path("./dataset/validation/0.25x")
    )


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
        config=dict(model_id="1_custom"),
        name=f"1_custom",
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

    train_dataset, validation_dataset = get_dataset()
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
    train()
    inference_onnxruntime()
    calc_and_print_PSNR()
