import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, EMNIST, KMNIST
from torchvision.transforms import ToTensor


class MnistDataset(Dataset):

    dataset_dir = 'mnist'

    def __init__(self, data_dir: str = 'data', mode: str = 'train', dataset: str = 'mnist') -> None:
        super().__init__()

        self.dataset = dataset
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.mode = mode
        self.prepare_data()

    def prepare_data(self) -> None:
        transform = ToTensor()
        if self.dataset == "emnist":
            if self.mode == 'train':
                self.dataset = EMNIST(self.dataset_dir, split='letters', download=True, train=True, transform=transform)
            elif self.mode == 'test':
                self.dataset = EMNIST(self.dataset_dir, split='letters', download=True, train=False, transform=transform)
            else:
                raise ValueError("Mode must be 'train' or 'test'")
            
            return

        DATASET = MNIST if self.dataset == 'mnist' \
                else FashionMNIST if self.dataset == 'fashion' \
                else KMNIST

        if self.mode == 'train':
            self.dataset = DATASET(self.dataset_dir, download=True, train=True, transform=transform)
        elif self.mode == 'test':
            self.dataset = DATASET(self.dataset_dir, download=True, train=False, transform=transform)
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)

        fourier_transform = np.fft.fft2(image)
        fourier_shifted = np.fft.fftshift(fourier_transform)
        magnitude = np.abs(fourier_shifted)
        magnitude = np.log1p(magnitude).astype(np.float32)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

        return image, magnitude, label


if __name__ == "__main__":
    dataset = MnistDataset(data_dir='data', mode='test', dataset='mnist')
    print(len(dataset))
    labels = []
    for i in range(64):
        image, magnitude, label = dataset[i]
        # print(image.shape, magnitude.shape, label)
        labels.append(label)
    print(labels)
    # print(image.shape, magnitude.shape, label)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image - Label: {label}")
    plt.imshow(image[0], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Magnitude Spectrum")
    plt.imshow(magnitude[0], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()