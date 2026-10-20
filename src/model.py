import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # Convolutional feature extraction layers
        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolution block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fourth convolution block
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fifth convolution block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        # Fully connected classification layers
        self.classifier = nn.Sequential(

            # Convert the feature maps into one long vector
            nn.Flatten(),

            # Learn high-level features
            nn.Linear(256 * 7 * 7, 512),

            nn.ReLU(),

            # Reduce overfitting
            nn.Dropout(0.5),

            # Final prediction for the three classes
            nn.Linear(512, num_classes)
        )


    def forward(self, x):

        # Extract visual features
        x = self.features(x)

        # Classify the extracted features
        x = self.classifier(x)

        return x


class TransferLearningModel(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze the pretrained layers
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Unfreeze layer4 for fine-tuning
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad = True

        # Replace the original ImageNet classifier
        # with our three bean-leaf classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)

