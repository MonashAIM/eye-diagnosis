import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def dataloader(dataset_filepath: str, regenerate_data: bool, saving_dump: bool = True, imagesize: int = 128, training_ratio: float = 0.8, percentage_of_folder_used: float = 1.0, transform="default", batch_size: int = 32):
    """
        this returns a dataloader, Uses pickle to save/load the data, if you are not changing your dataloader often
        but this can be disabled if you would like to avoid errors and regenerate data each time

        :param regenerate_data: load from datadump
        :param save_dump: save to datadump?

        # values here are for regenerating the dataloaders.
        :param dataset_filepath: filepath to image classes
        :param imagesize: size of tensors 512x512
        :param training_ratio: ratio 'training data to validation data' eg. 0.7 = 70%
        :param percentage_of_folder_used: ratio of images loaded.
        :param transform: custom transform.
        :param batch_size: batch size
        :return: training dataloader and the validation dataloader as a tuple
        """

    if regenerate_data:
        print("Generating the dataset from scratch")
        train_loader, val_loader = generate_dataloader_from_folder(dataset_filepath, imagesize, training_ratio, percentage_of_folder_used, transform, batch_size)
    else:
        # this regenerates the data from pickle, so you don't have to regenerate the data everytime
        print("Loading data from dump")
        train_loader, val_loader = load_dataloader_state("dump.pkl")

    # save dataloader_state
    if regenerate_data and saving_dump:
        print("now saving new dataset")
        save_dataloader_state("dump.pkl", train_loader, val_loader)
    return train_loader, val_loader



def generate_dataloader_from_folder(folder_path: str, imagesize: int = 128, training_ratio: float = 0.8, percentage_of_folder_used: float = 1.0, transform="default", batch_size: int = 32) -> (DataLoader, DataLoader):
    """
    Generates the Dataloaders from scratch from the files. (slower, but used when changes are needed)
    :param folder_path: filepath to image classes
    :param imagesize: size of tensors 512x512
    :param training_ratio: ratio 'training data to validation data' eg. 0.7 = 70%
    :param percentage_of_folder_used: ratio of images loaded.
    :param transform: custom transform.
    :param batch_size: batch size
    :return: training dataloader and the validation dataloader as a tuple
    """

    # This is just a Basic transform.
    # By Default just turns images into tensors of 128 and normalises the colour values.
    # I would recommend making your own, possible with random rotations.
    if transform == "default":
        transform = transforms.Compose([
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform)


    # I need to separate all the classes equally, it has to take an equal value from each class, so I have to do this.
    class_indices = {class_name: [] for class_name in dataset.classes}
    for idx, (image, label) in enumerate(dataset):
        class_name = dataset.classes[label]
        class_indices[class_name].append(idx)


    # Takes equal subset of "percentage_of_folder_used" from each data class"
    subset_indices = []
    for class_name, indices in class_indices.items():
        num_samples = int(len(indices) * percentage_of_folder_used)
        subset_indices.extend(np.random.choice(indices, num_samples, replace=False))

    subset_dataset = CustomSubset(dataset, subset_indices)

    # Define the ratio for training and validation
    train_size = int(training_ratio * len(subset_dataset))
    val_size = len(subset_dataset) - train_size

    # just randomly splitting the data.
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# this is so you don't have to load it each time from scratch.
def save_dataloader_state(file_path, train_loader, val_loader):
    """
    This uses pickle to create a serialised dump of the training and validation dataloaders.

    :param file_path: File/Filepath of dump.pkl
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :return: Void (saved dump.pkl)
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
    :param file_path: File/Filepath of dump.pkl
    :return: training dataloader, validation dataloader
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
