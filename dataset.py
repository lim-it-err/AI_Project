from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torch
from glob import glob
import torchvision.transforms as transforms
from PIL import Image


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
        PIL_image = Image.fromarray(image)      #JH: Added This Part
        if self.transform is not None:
            image = self.transform(PIL_image)
        return image, self.classes.index(self.label[idx])


if __name__ == "__main__":
    classes = ('MSIMUT', 'MSS')
    transform_train = transforms.Compose([
        transforms.RandomCrop(224, padding=18),
        # transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainloader = torch.utils.data.DataLoader(
        CRC_DataSet(glob('TCGA_DATA/CRC_TEST/*/*.png'),classes, transform_train),
        batch_size=4,
        shuffle=True
    )
    dataiter = iter(trainloader)
    image, label = dataiter.next()
    print(image[0].shape, label)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(batch_idx, inputs.shape, targets)
