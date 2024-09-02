import os
import random
import glob
from pathlib import Path
import json
import itertools

import numpy as np
import torch
import torchvision.transforms as T
import tqdm as tqdm
import wandb
import cv2
import PIL
import imagesize

PIL.Image.MAX_IMAGE_PIXELS = None

BATCH_SIZE = 64
NUM_WORKERS = 24
NUM_EPOCHS = 500_000
LEARNING_RATE = 1e-3
MODEL_NAME = "3ch_s3407_again"
LOG_IMAGE = False
# PRETRAINED = "./models/multires+flickr-67_psnr-28.29478.pth"
PRETRAINED = "./models/3ch_s420_soup.pth"
SEED = 3407


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


def calc_psnr(image1, image2):
    to_image = T.ToPILImage()
    image1 = cv2.cvtColor(
        (np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    image2 = cv2.cvtColor(
        (np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    return cv2.PSNR(image1, image2)


class ESPCN4x(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 4
        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            torch.nn.PReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=(3 * self.scale * self.scale),
                kernel_size=3,
                padding=1,
            ),
            torch.nn.PReLU(),
            torch.nn.PixelShuffle(self.scale),
        )

    def forward(self, batch):
        base = torch.nn.functional.interpolate(
            batch, scale_factor=self.scale, mode="bicubic"
        )
        x = self.nn(batch)
        x = torch.tanh(x)
        x = base + x
        x[x>1.0] = 1.0
        x[x<0.0] = 0.0
        return x


class DatasetGeneral(torch.utils.data.Dataset):
    def __init__(self, im_paths, preproc, n_im_per_epoch=2000, aug=None):
        self.im_paths = im_paths
        self.preproc = preproc
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
        im_small = T.Resize(
            (im_orig.size[0] // 4, im_orig.size[1] // 4),
            T.InterpolationMode.BICUBIC,
        )(im_orig.copy())
        im_orig = self.preproc(im_orig)
        im_small = self.preproc(im_small)
        return im_small, im_orig


class DatasetValSignate(torch.utils.data.Dataset):
    def __init__(
        self, images_small_Wild, preproc, small_id="0.25x", orig_id="original"
    ):
        self.im_small_paths = glob.glob(images_small_Wild)
        self.im_orig_paths = [x.replace(small_id, orig_id) for x in self.im_small_paths]
        self.preproc = preproc

    def __len__(self):
        return len(self.im_small_paths)

    def __getitem__(self, idx):
        im_small_p = self.im_small_paths[idx]
        im_small = PIL.Image.open(im_small_p).convert("RGB")
        im_orig_p = self.im_orig_paths[idx]
        im_orig = PIL.Image.open(im_orig_p).convert("RGB")
        return (self.preproc(im_small), self.preproc(im_orig))


def train(model, train_loader, val_loader, optimizer, scheduler, criterion):
    device = "cuda"
    model.to(device, memory_format=torch.channels_last)
    # model = torch.compile(model)
    #
    wandb.init(
        project="signate-superres-1374",
        entity="petrdvoracek",
        config=dict(model_id=MODEL_NAME),
        name=MODEL_NAME,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm.trange(NUM_EPOCHS, desc="EPOCH"):
        try:
            model.train()
            train_loss = []
            for low_res, high_res in tqdm.tqdm(
                train_loader, desc=f"EPOCH[{epoch}] TRAIN", total=len(train_loader)
            ):
                low_res = low_res.to(device, memory_format=torch.channels_last)
                high_res = high_res.to(device, memory_format=torch.channels_last)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    output = model(low_res)
                    loss = criterion(output, high_res)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss.append(loss.item() * low_res.size(0))
                # # skip psnr calculation for faster training
                # if should_calc_psnr:
                #     for img1, img2 in zip(output, high_res):
                #         train_psnr += calc_psnr(img1, img2)
            scheduler.step()

            model.eval()
            val_loss = []
            val_psnr = []
            best_val_psnr = 0.0
            with torch.no_grad():
                for low_res, high_res in tqdm.tqdm(
                    val_loader, desc=f"EPOCH[{epoch}] VALIDATION", total=len(val_loader)
                ):
                    low_res = low_res.to(device, memory_format=torch.channels_last)
                    high_res = high_res.to(device, memory_format=torch.channels_last)
                    with torch.amp.autocast("cuda"):
                        output = model(low_res)
                        loss = criterion(output, high_res)
                    val_loss.append(loss.item() * low_res.size(0))
                    for img1, img2 in zip(output, high_res):
                        val_psnr.append(calc_psnr(img1, img2))
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": sum(train_loss) / len(train_loss),
                    "val_loss": sum(val_loss) / len(val_loss),
                    "val_psnr": sum(val_psnr) / len(val_psnr),
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            if LOG_IMAGE:
                wandb.log(
                    {
                        "epoch": epoch,
                        "examples": wandb.Image(output[0:1], caption="asd"),
                    }
                )

            current_psnr = sum(val_psnr) / len(val_psnr)
            if current_psnr > best_val_psnr:
                best_val_psnr = current_psnr
                torch.save(
                    model.state_dict(),
                    f"models/{MODEL_NAME}-{epoch}_psnr-{current_psnr:.5f}.pth",
                )
        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    torch.save(model.state_dict(), f"models/{MODEL_NAME}-final.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        f"models/{MODEL_NAME}.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )


def main():
    seedme(SEED)

    in1k_val = glob.glob("/datasets/imagenet/val/*/*")
    in1k_val_large = [x for x in in1k_val if min(imagesize.get(x)) > 512]
    with open("./in1k_train_big.txt", "r") as f:
        in1k_train_large = json.load(f)
    datasets = dict(
        japan=glob.glob("/datasets/japan_160k/*/*"),
        superres=glob.glob("/datasets/superres/*/*"),
        trainims=glob.glob("./train/*"),
        flickr1=glob.glob("/datasets/unsplash_highres/*/*.bmp"),
        in1k_val=in1k_val_large,
        in1k_train=in1k_train_large,
    )

    train_paths = list(itertools.chain.from_iterable(datasets.values()))

    for k, v in datasets.items():
        print(f"{k}: {len(v)}")

    print(f"total images: {len(train_paths)}")

    preproc = T.ToTensor()

    augmentation = T.Compose(
        [
            # transforms.Pad(),
            T.RandomCrop((512, 512)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    )
    train_dataset = DatasetGeneral(
        train_paths, n_im_per_epoch=850 * 10, aug=augmentation, preproc=preproc
    )
    val_dataset = DatasetValSignate("./validation/0.25x/*", preproc=preproc)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True
    )

    model = ESPCN4x()
    if PRETRAINED:
        model.load_state_dict(torch.load(PRETRAINED))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5, last_epoch=-1
    )
    criterion = torch.nn.MSELoss()

    train(model, train_loader, val_loader, optimizer, scheduler, criterion)


if __name__ == "__main__":
    main()
