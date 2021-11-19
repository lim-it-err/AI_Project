from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torch

class CRC_DataSet(Dataset):
    # data_path_list - 이미지 path 전체 리스트
    # label - 이미지 ground truth
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = []
        for path in data_path_list:
            self.label.append(path.split('/')[-2])
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])
