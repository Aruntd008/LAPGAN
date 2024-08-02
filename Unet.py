import helpMe
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from PIL import Image, ImageFilter

from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetGenerator, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.bottleneck = nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1)

        # Downsampling part
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Upsampling part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)



class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1)
        self.conv_layers = nn.ModuleList()
        
        in_features = features[0]
        for feature in features[1:]:
            self.conv_layers.append(self._block(in_features, feature, stride=2))
            in_features = feature
        
        self.final_conv = nn.Conv2d(in_features, 1, kernel_size=2, stride=1, padding=0)
        
    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(x))
        for layer in self.conv_layers:
            x = layer(x)
            # print(x.shape)
            
        return torch.sigmoid(self.final_conv(x))
    
    
class CUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, image_size = 32, num_classes=10, features=[64, 128, 256, 512]):
        super(CUNetGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)  # Assuming square images
        self.in_channels = in_channels

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.bottleneck = nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1)
        
        
        # self.conv1 = nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1)
        
        self.downs.append(self.conv_block(in_channels+1, features[0]))
        in_channels = features[0]
        # Downsampling part
        for feature in features[1:]:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Upsampling part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, labels):
        # Embedding labels
        label_embeddings = self.label_embedding(labels).view(labels.size(0), self.in_channels, x.size(2), x.size(3))
        x = torch.cat([x, label_embeddings], dim=1)
        
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(  x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

class CDiscriminator(nn.Module):
    def __init__(self, in_channels=1, image_size=32, num_classes=10, features=[64, 128, 256, 512]):
        super(CDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)  # Assuming square images
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1)
        self.conv_layers = nn.ModuleList()
        
        in_features = features[0]
        for feature in features[1:]:
            self.conv_layers.append(self._block(in_features, feature, stride=2))
            in_features = feature
        
        self.final_conv = nn.Conv2d(in_features, 1, kernel_size=2, stride=1, padding=0)
        
    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x, labels):
        label_embeddings = self.label_embedding(labels).view(labels.size(0), self.in_channels, x.size(2), x.size(3))
        x = torch.cat([x, label_embeddings], dim=1)
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(x))
        for layer in self.conv_layers:
            x = layer(x)
            
        return torch.sigmoid(self.final_conv(x))    


# class Generator_L(nn.Module):
#     def __init__(self, channels=1):
#         super(Generator_L, self).__init__()
#         self.channels = channels
#         self.fc1 = nn.Linear(100, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 1024)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.fc4 = nn.Linear(1024, 8 * 8 * channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = torch.tanh(self.fc4(x))
#         x = x.view(-1, self.channels, 8, 8)
#         return x
    
# class Discriminator_L(nn.Module):
#     def __init__(self, channels =1):
#         super(Discriminator_L, self).__init__()
#         self.channels = channels
#         self.fc1 = nn.Linear(channels * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)

#     def forward(self, x):
#         x = x.view(-1, self.channels * 8 * 8)
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = torch.sigmoid(self.fc3(x))
#         return x



# class Generator_L(nn.Module):
#     def __init__(self, noise_dim=100, image_size=8, hidden_dim=128):
#         super(Generator_L, self).__init__()
#         self.noise_dim = noise_dim
#         self.image_size = image_size
#         self.hidden_dim = hidden_dim

#         self.model = nn.Sequential(
#             nn.Linear(noise_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim*2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim*2, image_size*image_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.model(x).view(-1, 1, self.image_size, self.image_size)


# # Discriminator Model
# class Discriminator_L(nn.Module):
#     def __init__(self, image_size=8, hidden_dim=128):
#         super(Discriminator_L, self).__init__()
#         self.image_size = image_size
#         self.hidden_dim = hidden_dim

#         self.model = nn.Sequential(
#             nn.Linear(image_size*image_size, hidden_dim*2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x.view(-1, self.image_size*self.image_size))


class Generator_L(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, image_size=8, hidden_dim=128):
        super(Generator_L, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, image_size*image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        return self.model(gen_input).view(-1, 1, self.image_size, self.image_size)

# Discriminator Model
class Discriminator_L(nn.Module):
    def __init__(self, num_classes=10, image_size=8, hidden_dim=128):
        super(Discriminator_L, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(image_size*image_size + num_classes, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img.view(img.size(0), -1), label_embedding), -1)
        return self.model(d_in)
    
    
    
    
class Generator_L2(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, image_size=8, hidden_dim=128):
        super(Generator_L2, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, image_size*image_size*3),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        return self.model(gen_input).view(-1, 3, self.image_size, self.image_size)
    
    
class Discriminator_L2(nn.Module):
    def __init__(self, num_classes=10, image_size=8, hidden_dim=128):
        super(Discriminator_L2, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(image_size*image_size*3 + num_classes, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img.view(img.size(0), -1), label_embedding), -1)
        return self.model(d_in)