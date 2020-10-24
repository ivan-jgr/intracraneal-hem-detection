import torch
import cv2
import settings
import pandas as pd

from os.path import join
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):

    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Arguments
        ---------
        root_dir:       path del folder con las imagenes png
        dataset_type:   train o val dataset
        transform:      transformaciones que se deben aplicar a las imagenes
        """
        super(ImageDataset, self).__init__()

        self.images_path = join(root_dir, 'img')
        self.dataframe = pd.read_csv(join(root_dir, 'dicom_info.csv'))

        with open(join(root_dir, dataset_type + '.txt'), 'r') as f:
            patients = f.read().splitlines()

        self.dataframe = self.dataframe[self.dataframe['study_instance_uid'].isin(patients)]

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        filename = row.filename
        labels = torch.from_numpy(row['any':'subdural'].values)

        image = cv2.imread(join(self.images_path, filename), 0)

        if self.transform is not None:
            image = self.transform(image=image)

        return image, labels


def get_data_loaders(train_transform, val_transform):
    train_dataset = ImageDataset('../data', 'train', train_transform)
    val_dataset = ImageDataset('../data', 'val', val_transform)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=settings.batch_size,
                                   shuffle=True,
                                   num_workers=settings.workers,
                                   pin_memory=True,
                                   drop_last=True)

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=settings.batch_size,
                                 shuffle=False,
                                 num_workers=settings.workers,
                                 pin_memory=True,
                                 drop_last=False)

    data_loaders = {'train': train_data_loader, 'val': val_data_loader}

    return data_loaders
