from torch.utils.data.dataset import Dataset
from PIL import Image


class ClassDataset(Dataset):

    def __init__(self, classes, train=True, root="", ds=None, transform=None, target_transform=None, save_path=""):
        if not (root or ds):
            raise ValueError("ClassDataset requires either a root dir or a dataset to reduce")

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if self.train:
            class_indices = [ind for ind, target_class in enumerate(ds.train_labels) if target_class in classes]
            self.data = ds.train_data[class_indices]
            self.labels = ds.train_labels[class_indices]
        else:
            class_indices = [ind for ind, target_class in enumerate(ds.test_labels) if target_class in classes]
            self.data = ds.test_data[class_indices]
            self.labels = ds.test_labels[class_indices]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
