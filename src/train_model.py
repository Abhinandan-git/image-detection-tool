# train_model.py
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from image_preprocessing import create_dataloader
from model import create_model
from data_exploration import load_data

def train_model(train_df, num_classes, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(num_classes).to(device)
    dataloader = create_dataloader(train_df, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, torch.zeros(outputs.size(0)).long().to(device))  # Dummy target for example
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), 'model.pth')
    print("Model training complete and saved as model.pth")

if __name__ == "__main__":
    train_df, _ = load_data('dataset/train.csv', 'dataset/test.csv')
    train_model(train_df, num_classes=10)  # Adjust num_classes as needed
