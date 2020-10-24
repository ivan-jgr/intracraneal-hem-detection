import torch
import settings
import copy
import time
import torch.optim as optim

from model.resnet import ResnetModel
from torch.optim import lr_scheduler
from dataloader.dataloader import get_data_loaders
from torchvision import transforms


import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloaders, num_epochs=50):

    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'val']}

    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=settings.lr, momentum=settings.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=settings.step_size, gamma=settings.gamma)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    tic = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - tic
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)

    torch.save(best_model_wts, 'checkpoints/best_model.pth')

    return model


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataloaders = get_data_loaders(train_transform, val_transform)
    model = ResnetModel(2)

    train(model, dataloaders, num_epochs=settings.epochs)
