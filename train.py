import copy

import numpy as np
import torch
from criterion import HeScho
from DIBCO import DIBCO
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from unet import DeepOtsu


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(270),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {
        x: DIBCO("./dataset/DIBCO", transform=data_transforms[x])
        for x in ('train', 'val')
    }

    model = DeepOtsu(3)

    criterion = HeScho()

    n_epochs = 100
    lr = 0.01
    batch_size = 16
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    dataloaders = {'train': train_loader, 'val': validation_loader}

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

            loader = dataloaders[phase]
            n_total_steps = len(loader)

            for i, (img, gt) in enumerate(loader):
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

                running_loss += loss.item() * img.size(0)
                running_corrects += torch.sum(output == gt.data)

            if phase == 'train':
                scheduler.step()

                print(
                    f"Epoch: {epoch + 1}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss: {loss.item():.4f}"
                )

            epoch_loss = running_loss / n_total_steps
            epoch_acc = running_corrects.double() / n_total_steps

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './weight.pth')


if __name__ == "__main__":
    main()