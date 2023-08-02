import torchvision
import torch
class Aug_toTensor:
    " process the data to tensor"
    def __init__(self ):
        self.train_transform_1 = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
            
        ]
        self.train_transform_2 = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        
        self.test_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform_1(x), self.train_transform_2(x)
class Aug_toTensor2:
    " process the data to tensor"
    def __init__(self ):
        self.train_transform_1 = [
            torchvision.transforms.ToTensor(),
            
        ]
        self.train_transform_2 = [
            torchvision.transforms.ToTensor(),
        ]
        
        self.test_transform = [
            torchvision.transforms.ToTensor(),
        ]
        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform_1(x), self.train_transform_2(x)

class Aug_GrayToGray3:
    '''
        该方法用于将单通道图像变成多通道图像灰度图像
    '''
    def __init__(self ):
        self.train_transform_1 = [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.train_transform_2 = [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.test_transform = [
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        
        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform_1(x), self.train_transform_2(x)

class Aug_RGBtoGray:
    '''
    该方法用于将彩色图像变成单通道图像
    '''
    def __init__(self ):
        self.train_transform_1 = [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.train_transform_2 = [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.test_transform = [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float64)
        ]
        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform_1(x), self.train_transform_2(x)

