# evaluate_model.py
import torch
import pandas as pd
from image_preprocessing import preprocess_image
from model import create_model

def evaluate_model(test_df, model_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(num_classes=10)  # Adjust num_classes as needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions = []
    for _, row in test_df.iterrows():
        image_url = row['image_link']
        img_tensor = preprocess_image(image_url).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    test_df['prediction'] = predictions
    test_df.to_csv('predictions.csv', index=False)
    print("Evaluations complete. Predictions saved as predictions.csv")

if __name__ == "__main__":
    _, test_df = load_data('dataset/train.csv', 'dataset/test.csv')
    evaluate_model(test_df)
