import numpy as np
from sklearn.model_selection import train_test_split
import torch


base_dir = "kaggleData/"
TRAIN_IMAGES_LOC = base_dir+"train_images.npy"
TRAIN_LABELS_LOC = base_dir+"train_labels.npy"
TEST_IMAGES_LOC = base_dir+"test_images.npy"

class KaggleData:

    def __init__(self, test_split=.7):
        self.images = np.load(TRAIN_IMAGES_LOC)
        self.labels = np.load(TRAIN_LABELS_LOC)
        self.hidden_test_images = np.load(TEST_IMAGES_LOC)
        # names to fit into ClassDataset
        self.train_data, self.test_data, self.train_labels, self.test_labels = self.__get_train_validation_data(test_split)

        self.train_data = torch.tensor(self.train_data)
        self.test_data = torch.tensor(self.test_data)
        self.train_labels = torch.tensor(self.train_labels).long()
        self.test_labels = torch.tensor(self.test_labels).long()


    def reformat_data(self, data):
        # formatted_data = []
        # for item in data:
        #     formatted_data.append(torch.tensor(item.reshape(28, 28)))
        return data.reshape(-1, 28, 28)
        # return torch.tensor(np.array(formatted_data))

    def get_full_data(self):
        return self.images, self.labels

    def __get_train_validation_data(self, test_split):
        return train_test_split(self.reformat_data(self.images), self.labels, test_size=test_split)

    def get_hidden_test_data(self):
        return self.hidden_test_images

