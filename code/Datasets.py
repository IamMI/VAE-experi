"""
Load ImageNet datasets and add noise to images
"""
import os
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision import datasets


class ImageNetdatasets():
    def __init__(self, 
                 dataset_name="imagenet-1k", 
                 data_path="/megadisk/DATASETS/imagenet/"):
        """
        Initialize the ImageNet datasets
        :param dataset_name: 数据集名称
        :param data_path: 路径
        """
        self.dataset_name = dataset_name
        self.data_path = data_path    
        
    def load_dataset(self, Option=None):
        """
        Load the dataset
        """
        if Option not in ["train", "val", "test"]:
            print("Please specify the dataset: train or val or test")
            exit()
        
        datapath = os.path.join(self.data_path, Option)
        dataset = datasets.ImageFolder(datapath, 
                                        transform=transforms.Compose([
                                            transforms.Resize((256,256)),
                                            transforms.ToTensor(),
        ]))
        
        if Option=='train':
            sample_radio = 0.01
            dataset = Subset(dataset, range(int(len(dataset)*sample_radio)))
        elif Option=='val':
            sample_radio = 0.01
            dataset = Subset(dataset, range(int(len(dataset)*sample_radio)))
        
        return dataset 
        

if __name__ == "__main__":
    dataset = ImageNetdatasets().load_dataset('train')
    
    