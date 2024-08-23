# %%
import os
import random
import glob
import copy
import time

import numpy as np
import torch
import torchvision.transforms as T
import tqdm as tqdm
import wandb
import cv2
import PIL
import imagesize
import click
import kornia.augmentation as KA
import kornia as K


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
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=5, padding=2
        )
        self.act = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )
        self.conv_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.conv_4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=(3 * self.scale * self.scale),
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(self.scale)

    def forward(self, batch):
        base = torch.nn.functional.interpolate(
            batch, scale_factor=self.scale, mode="bicubic"
        )

        x = self.act(self.conv_1(batch))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = base + x

        X_out = torch.clip(x, 0.0, 1.0)
        return X_out


class DatasetGeneral(torch.utils.data.Dataset):
    def __init__(self, im_paths2d, preproc, n_im_per_epoch=2000, aug=None):
        self.im_paths2d = im_paths2d
        self.preproc = preproc
        self.aug = aug
        self.len = n_im_per_epoch

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        # equally balance datasets
        ds_idx = random.randint(0, len(self.im_paths2d) - 1)
        ds = self.im_paths2d[ds_idx]
        im_idx = random.randint(0, len(ds) - 1)
        p = ds[im_idx]
        im_orig = PIL.Image.open(p)
        if self.aug is not None:
            im_orig = self.aug(im_orig)
        im_small = T.Resize(
            (im_orig.size[0] // 4, im_orig.size[1] // 4),
            random.choice(
                [
                    T.InterpolationMode.BICUBIC,
                    # transforms.InterpolationMode.BILINEAR,
                    # transforms.InterpolationMode.NEAREST_EXACT,
                    # transforms.InterpolationMode.NEAREST,
                    # transforms.InterpolationMode.LANCZOS,
                    # transforms.InterpolationMode.HAMMING,
                ]
            ),
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
        im_small = PIL.Image.open(im_small_p)
        im_orig_p = self.im_orig_paths[idx]
        im_orig = PIL.Image.open(im_orig_p)
        return (self.preproc(im_small), self.preproc(im_orig))


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    name,
    log_image,
):
    device = "cuda"
    model.to(device, memory_format=torch.channels_last)
    # model = torch.compile(model)
    #
    wandb.init(
        project="signate-superres-1374",
        entity="petrdvoracek",
        config=dict(model_id=name),
        name=name,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm.trange(num_epochs, desc="EPOCH"):
        always_calc_psnr = epoch > epoch - 10

        model.train()
        train_loss = 0.0
        train_psnr = []
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
            train_loss += loss.item() * low_res.size(0)
            if always_calc_psnr or (random.random() < 0.2):
                for img1, img2 in zip(output, high_res):
                    train_psnr.append(calc_psnr(img1, img2))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_psnr = []
        with torch.no_grad():
            for low_res, high_res in tqdm.tqdm(
                val_loader, desc=f"EPOCH[{epoch}] VALIDATION", total=len(val_loader)
            ):
                low_res = low_res.to(device, memory_format=torch.channels_last)
                high_res = high_res.to(device, memory_format=torch.channels_last)
                with torch.amp.autocast("cuda"):
                    output = model(low_res)
                    loss = criterion(output, high_res)
                val_loss += loss.item() * low_res.size(0)
                if always_calc_psnr or (random.random() < 0.2):
                    for img1, img2 in zip(output, high_res):
                        val_psnr.append(calc_psnr(img1, img2))

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader.dataset),
                "val_loss": val_loss / len(val_loader.dataset),
                "train_psnr": sum(train_psnr) / len(train_psnr),
                "val_psnr": sum(val_psnr) / len(val_psnr),
            }
        )
        if log_image:
            wandb.log(
                {
                    "epoch": epoch,
                    "examples": wandb.Image(output[0:1], caption="asd"),
                }
            )

        if epoch % 10 == 0 or epoch > num_epochs - 10:
            torch.save(model.state_dict(), f"models/{name}-{epoch}.pth")

    torch.save(model.state_dict(), f"models/{name}-final.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        f"models/{name}.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )

class Augmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Sequential(
                KA.RandomCrop(512),
                KA.RandomHorizontalFlip(),
                KA.RandomVerticalFlip(),
        )
    def forward(self, x):
        return self.t(x)

class PreProcess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float()

@click.command()
@click.argument("NAME")
@click.option("--batch-size", default=64)
@click.option("--num-workers", default=8)
@click.option("--num-epochs", default=500)
@click.option("--lr", default=1e-3)
@click.option("--log-image", is_flag=True)
def main(name, batch_size, num_workers, num_epochs, lr, log_image):
    seedme(42)

    # train_dataset, validation_dataset = get_dataset()
    in1k_val = glob.glob("/datasets/imagenet/val/*/*")
    in1k_val_large = [x for x in in1k_val if min(imagesize.get(x)) > 512]
    in1k_train = glob.glob("/datasets/imagenet/train/*/*")
    in1k_train_large = [x for x in in1k_train if min(imagesize.get(x)) > 512]
    train_paths = [
        glob.glob("/datasets/japan_160k/*/*"),
        glob.glob("/datasets/superres/*/*"),
        glob.glob("../train/*"),
        # glob.glob("/dev/shm/superres/*/*"),
        # glob.glob("/dev/shm/train/*"),
        in1k_val_large,
        in1k_train_large,
    ]
    # just flattening the list
    print(f"n images: {len([el for sublist in train_paths for el in sublist])}")

    preproc = T.ToTensor()
    augmentation = T.Compose(
        [
            T.RandomCrop(size=512),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    )

    train_dataset = DatasetGeneral(
        train_paths, preproc=preproc, aug=augmentation, n_im_per_epoch=850 * 10
    )
    val_dataset = DatasetValSignate("../validation/0.25x/*", preproc=preproc)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = ESPCN4x()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.3 * num_epochs),
            int(0.5 * num_epochs),
            int(0.65 * num_epochs),
            int(0.8 * num_epochs),
            int(0.9 * num_epochs),
            int(0.95 * num_epochs),
            int(0.97 * num_epochs),
        ],
        gamma=0.7,
    )
    criterion = torch.nn.MSELoss()

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        num_epochs,
        name,
        log_image,
    )


if __name__ == "__main__":
    main()
