import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def dataloader(dataset_filepath: str, regenerate_data: bool, saving_dump: bool = True, imagesize: int = 128, training_ratio: float = 0.8, percentage_of_folder_used: float = 1.0, transform="default", batch_size: int = 32):
    """
        this returns a dataloader, Use's pickle to save/load the data, if you are not changing your dataloader often
        but this can be disabled if you would like to avoid errors and regnerate data each time
        :param regenerate_data: whether you would like to load from files or from dump (pre-generated)
        :param save_dump: whether you would like to skip saving the data when generating from scratch.

        # values here are for regenerating the dataloaders.
        :param dataset_filepath: the data folder (eg. the directory above where 'cataracts, normal ... etc" are stored)
        :param imagesize: size of tensors
        :param training_ratio: this is the ratio of 'training data to validation data' eg. 0.7 = 70% which would mean that 30% is validation data and 70% is used for training
        :param percentage_of_folder_used: this is the amount of files we load from each folder, eg. 10% means we load 10% of the cataracts, normal and so on.
        :param transform: this is by default set to be the normal transform I find online, but if you want to change it you can post the values here.
        :param batch_size: this is batch size the images are loaded in
        :return: training data loader and the validation dataloader as a tuple
        """

    if regenerate_data:
        print("Generating the dataset from scratch")
        train_loader, val_loader = generate_dataloader_from_folder(dataset_filepath, imagesize, training_ratio, percentage_of_folder_used, transform, batch_size)
    else:
        # this regenerates the data from pickle, so you don't have to regenerate the data everytime
        print("Loading data from dump")
        train_loader, val_loader = load_dataloader_state("dump.pkl")

    if regenerate_data and saving_dump:
        print("now saving new dataset")
        # save dataloader_state
        save_dataloader_state("dump.pkl", train_loader, val_loader)
    return train_loader, val_loader



def generate_dataloader_from_folder(folder_path: str, imagesize: int = 128, training_ratio: float = 0.8, percentage_of_folder_used: float = 1.0, transform="default", batch_size: int = 32) -> (DataLoader, DataLoader):
    """
    Generates the Dataloaders from scratch from the files. (slower, but used when changes are needed)
    :param folder_path: the data folder (eg. the directory above where 'cataracts, normal ... etc" are stored)
    :param imagesize: this is the initial size of the images before being pooled.
    :param training_ratio: this is the ratio of 'training data to validation data' eg. 0.7 = 70% which would mean that 30% is validation data and 70% is used for training
    :param percentage_of_folder_used: this is the amount of files we load from each folder, eg. 10% means we load 10% of the cataracts, normal and so on.
    :param transform: this is by default set to be the normal transform I find online, but if you want to change it you can post the values here.
    :param batch_size: this is batch size the images are loaded in
    :return: training data loader and the validation dataloader as a tuple
    """
    if transform == "default":
        transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load the dataset
    original_dataset = datasets.ImageFolder(root=folder_path, transform=transform)

    # Create indices for each class
    class_indices = {class_name: [] for class_name in original_dataset.classes}
    for idx, (image, label) in enumerate(original_dataset):
        class_name = original_dataset.classes[label]
        class_indices[class_name].append(idx)

    # Sample a percentage from each class
    subset_indices = []
    for class_name, indices in class_indices.items():
        num_samples = int(len(indices) * percentage_of_folder_used)
        subset_indices.extend(np.random.choice(indices, num_samples, replace=False))

    # Create a subset dataset
    subset_dataset = CustomSubset(original_dataset, subset_indices)

    # Define the ratio for training and validation
    train_size = int(training_ratio * len(subset_dataset))
    val_size = len(subset_dataset) - train_size

    # this is the part which actually splits the training data from the validation data.
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

    # Create DataLoader for training
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Create DataLoader for validation
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# this is so you don't have to load it each time from scratch.
def save_dataloader_state(file_path, train_loader, val_loader):
    """
    This uses pickle to create a serialised dump of the training and validation dataloaders.

    :param file_path: This is the file where it will be saved (remember to include .pkl at the end)
    :param train_loader: This is current dataloader for training data.
    :param val_loader: This is current dataloader for validation data.
    :return: saved file.
    """
    state = {
        'train_indices': train_loader,
        'val_indices': val_loader,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(state, f)
        print("dataloader saved successfully.")


def load_dataloader_state(file_path):
    """
    :param file_path: this is the path/file we are loading from eg. data/dump.pkl
    :return: returns the training dataloader and the validation data loader
    """
    with open(file_path, 'rb') as f:
        dataloaders = pickle.load(f)
        train_loader = dataloaders['train_indices']
        val_loader = dataloaders['val_indices']
    return train_loader, val_loader




class CustomSubset(Dataset):
    """
    this is just a small fix to make sure we can load a fixed subset of the data of each category of folder instead
    of 10% of all, which may lead to uneven ratios for different classes.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
