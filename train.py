import logging

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary
from torchmetrics.functional import accuracy, f1, psnr

from criterion import HeScho
from dataset import DIBCO
from deepotsu import DeepOtsu
from logger import setup_logger
from transform import (Compose, Grayscale, Normalize, RandomCrop, RandomInvert,
                       RandomRotation, RandomScale, ToTensor)


def main():

    logger = logging.getLogger('Train')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger("./log/log.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = {
        'train':
        Compose([
            Grayscale(),
            RandomInvert(1),
            RandomScale(.75, 1.5, .25),
            RandomCrop(256, 256,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       True),
            RandomRotation(270),
            ToTensor(),
        ]),
        'val':
        Compose([
            Grayscale(),
            RandomInvert(1),
            RandomScale(1, 1, 0),
            RandomCrop(256, 256,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       False),
            ToTensor(),
        ]),
    }

    datasets = {
        x: DIBCO("./dataset/DIBCO", transform=data_transforms[x])
        for x in ('train', 'val')
    }

    model = DeepOtsu(1, num_block=2)
    try:
        model.load_state_dict(torch.load('./weights.pth'))
    except:
        pass
    model.to(device)
    summary(model, (1, 1, 256, 256))

    criterion = HeScho()

    n_epochs = 1000
    batch_size = 4
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(datasets['train'])
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(datasets['train'],
                              batch_size=batch_size,
                              sampler=train_sampler)
    validation_loader = DataLoader(datasets['val'],
                                   batch_size=batch_size,
                                   sampler=valid_sampler)

    optimizer = torch.optim.Adam(model.parameters())
    dataloaders = {'train': train_loader, 'val': validation_loader}

    best_acc = 0.0
    best_f1 = 0.0
    best_psnr = 0.0

    # Training loop
    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            f1_score = 0.0
            psnr_score = 0.0
            acc_score = 0.0

            loader = dataloaders[phase]
            n_total_steps = len(loader)

            for i, (img, gt, fnames) in enumerate(loader):
                img = img.to(device)
                gt = gt.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img)
                    loss = criterion(output, gt)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * img.size(0)
                f1_score += f1(output[-1], gt.data.type(torch.uint8))
                acc_score += accuracy(output[-1], gt.data.type(torch.uint8))
                psnr_score += psnr(output[-1], gt.data)

                if (epoch + 1) % 10 == 0:
                    out = output[-1][0]
                    fname = fnames[0].split('/')[-1].split(".")[0]
                    fname = '/'.join(("./log/debug_imgs", fname + ".jpg"))
                    torchvision.utils.save_image(out, fname)

            epoch_loss = running_loss / n_total_steps
            epoch_acc = acc_score / n_total_steps
            epoch_f1 = f1_score / n_total_steps
            epoch_psnr = psnr_score / n_total_steps

            msg = "Epoch: {}/{} \t {} \t Loss: {:.4f} F1: {:.4f} Acc: {:.4f}, PSNR: {:.4f}".format(
                epoch + 1, n_epochs, "Train" if phase == 'train' else "Val",
                epoch_loss, epoch_f1, epoch_acc, epoch_psnr)
            logger.info(msg)

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), './weights.pth')
                msg = "Saved best model. F1 Score: {:.4f}".format(epoch_f1)
                logger.info(msg)

    print('Best val F1 Score: {:.4f}'.format(best_f1))


if __name__ == "__main__":
    main()
