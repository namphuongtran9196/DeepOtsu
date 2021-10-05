import copy
import logging

#  import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from criterion import HeScho
from DIBCO import DIBCO
from logger import setup_logger
from transform import (Compose, Normalize, RandomCrop, RandomRotation,
                       RandomScale, ToTensor)
from unet import DeepOtsu


def main():

    logger = logging.getLogger('Train')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger("./log/log.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = {
        'train':
        Compose([
            RandomScale(.75, 1.5, .25),
            RandomCrop(256, 256,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       True),
            RandomRotation(270),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':
        Compose([
            RandomScale(1, 1, 0),
            RandomCrop(256, 256,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       False),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {
        x: DIBCO("./dataset/DIBCO", transform=data_transforms[x])
        for x in ('train', 'val')
    }

    model = DeepOtsu(3, 4)
    model.to(device)

    criterion = HeScho()

    n_epochs = 1000
    lr = 0.01
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
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    dataloaders = {'train': train_loader, 'val': validation_loader}

    best_acc = 0.0
    val_acc_history = []

    model.load_state_dict(torch.load('./weights.pth'))

    # Training loop
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            loader = dataloaders[phase]
            n_total_steps = len(loader)

            for i, (img, gt) in enumerate(loader):
                #  f, axarr = plt.subplots(1, 2)
                #  axarr[0].imshow(img[0].permute(1, 2, 0))
                #  axarr[1].imshow(gt[0].permute(1, 2, 0))
                #  plt.show()
                img = img.to(device)
                gt = gt.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img)
                    loss = criterion(output, gt)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(output == gt.data)
                running_total += gt.numel()
                if phase == 'train':
                    msg = f"Epoch: {epoch + 1}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss: {loss.item():.4f}, acc: {running_corrects / running_total:.4f}"
                    logger.info(msg)

            # if phase == 'train':
            #     scheduler.step()
            #     print(
            #         f"Epoch: {epoch + 1}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss: {loss.item():.4f}"
            #     )

            epoch_loss = running_loss / n_total_steps
            epoch_acc = running_corrects / running_total / n_total_steps

            msg = f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            logger.info(msg)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './weights.pth')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)


if __name__ == "__main__":
    main()
