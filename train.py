import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from criterion import CrossEntropyLoss, HeScho
from DIBCO import DIBCO
from logger import setup_logger
from transform import (Compose, Grayscale, Normalize, RandomCrop,
                       RandomRotation, RandomScale, ToTensor)
from unet import DeepOtsu


def main():

    logger = logging.getLogger('Train')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger("./log/log.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = {
        'train':
        Compose([
            Grayscale(),
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
            Grayscale(),
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

    model = DeepOtsu(1, num_block=2)
    try:
        model.load_state_dict(torch.load('./weights.pth'))
    except:
        pass
    model.to(device)

    criterion = CrossEntropyLoss()
    #     criterion = HeScho()

    n_epochs = 1000
    batch_size = 8
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
    val_acc_history = []

    # Training loop
    for epoch in range(n_epochs):
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

            for i, (img, gt, _) in enumerate(loader):
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
                running_corrects += torch.sum(output[-1] == gt.data)
                running_total += gt.numel()

            epoch_loss = running_loss / n_total_steps
            epoch_acc = running_corrects / running_total / n_total_steps

            msg = "Epoch: {}/{} \t {} \t Loss: {:.4f} Acc: {:.3e}".format(
                epoch + 1, n_epochs, "Train" if phase == 'train' else "Val",
                epoch_loss, epoch_acc)
            logger.info(msg)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './weights.pth')
                msg = "Saved best model. Accuracy: {:.3e}".format(epoch_acc)
                logger.info(msg)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    print('Best val Acc: {:4f}'.format(best_acc))


#     model.load_state_dict(torch.load('./weights.pth'))

if __name__ == "__main__":
    main()
