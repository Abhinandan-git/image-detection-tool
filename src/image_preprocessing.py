# image_preprocessing.py
from PIL import Image
import torchvision.transforms as transforms
import requests
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class ProductImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_url = self.dataframe.iloc[idx]['image_link']
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure the image is in RGB mode
    img_tensor = preprocess(img)
    return img_tensor

def create_dataloader(dataframe, batch_size=32):
    dataset = ProductImageDataset(dataframe, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    # Example usage
    from data_exploration import load_data
    train_df, _ = load_data('dataset/train.csv', 'dataset/test.csv')
    dataloader = create_dataloader(train_df)

    # Display batch shape
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        break
