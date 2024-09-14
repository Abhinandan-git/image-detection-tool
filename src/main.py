# main.py
from data_exploration import load_data, show_sample_images
from image_preprocessing import create_dataloader
from model import create_model
from train_model import train_model

def main():
    # Step 1: Load and Explore Data
    print("Loading and exploring data...")
    train_df, test_df = load_data('./dataset/train.csv', 'dataset/test.csv')
    show_sample_images(train_df)

    # Step 2: Preprocess Images (Create DataLoader)
    print("Creating DataLoader for training data...")
    dataloader = create_dataloader(train_df)

    # Step 3: Train the Model
    print("Training the model...")
    num_classes = 10  # Adjust based on your dataset and task
    model = create_model(num_classes=num_classes)
    train_model(train_df, num_classes=num_classes, epochs=10, batch_size=32, learning_rate=0.001)

if __name__ == "__main__":
    main()
