# model.py
import torch
import torchvision.models as models
import torch.nn as nn

class EntityExtractor(nn.Module):
    def __init__(self, num_classes):
        super(EntityExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def create_model(num_classes):
    model = EntityExtractor(num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = create_model(num_classes=1)  # Example for binary classification
    print(model)
