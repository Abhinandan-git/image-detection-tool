# data_exploration.py
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def load_data(train_path, test_path):
  train_df = pd.read_csv(train_path)
  test_df = pd.read_csv(test_path)
  return train_df, test_df

def show_sample_images(df, n_samples=5):
  fig, axes = plt.subplots(1, n_samples, figsize=(20, 5))
  for i in range(n_samples):
    response = requests.get(df.iloc[i]['image_link'])
    img = Image.open(BytesIO(response.content))
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(df.iloc[i]['entity_name'])
  plt.show()

if __name__ == "__main__":
  train_df, test_df = load_data('dataset/train.csv', 'dataset/test.csv')
  show_sample_images(train_df)
