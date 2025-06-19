# Automatic Clothing Classification and Text-Based Image Search for E-Commerce

## Project Description

E-commerce platforms offer a wide range of fashion products, making it increasingly difficult for users to locate specific items efficiently. This project addresses that challenge by designing and evaluating deep learning models that can classify fashion images into predefined categories and retrieve relevant images based on textual descriptions.

We implemented and compared the performance of four models:
- A custom Convolutional Neural Network (CNN)
- A pre-trained ResNet18 model
- A Vision Transformer (ViT) implemented from scratch
- OpenAI’s CLIP model for zero-shot image–text matching

## Dataset

The project uses the publicly available Fashion Images dataset by Vikash Raj on Kaggle:  
https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

From the original dataset of 2906 images, we filtered 12 distinct categories:
- Tshirts, Casual Shoes, Heels, Sport Shoes, Tops, Flip Flops, Flats, Sandals, Shorts, Shirts, Formal Shoes, Dresses

The dataset was divided into:
- Training set: 70% (1870 images)
- Validation set: 20% (535 images)
- Test set: 10% (268 images)

## Model Implementation

Each model was implemented and trained using the PyTorch framework. Key architectural and training parameters were as follows:

- **CNN:** Custom architecture with 2 convolutional layers, ReLU activation, max pooling, dropout, and fully connected layers.  
- **ResNet18:** Pre-trained on ImageNet, fine-tuned on the fashion dataset.
- **ViT:** Built manually with patch embeddings, transformer blocks, and classification head. Includes dropout and positional encoding.
- **CLIP:** Used for zero-shot retrieval without additional training, using ViT-B/32 variant.

## Evaluation Metrics

The models were evaluated using the following metrics:
- Test Accuracy
- Recall
- F1-Score
- Confusion Matrix
- Top-k prediction outputs

### Performance Summary

| Model  | Test Accuracy | Recall | F1-Score |
|--------|---------------|--------|----------|
| CNN    | 0.79          | 0.74   | 0.74     |
| ResNet | 0.75          | 0.80   | 0.77     |
| ViT    | 0.46          | 0.32   | 0.15     |

CLIP achieved successful text-to-image matching for dominant and visually distinct categories, demonstrating potential for future development in retrieval-based fashion search.

## Tools and Libraries

- Python 3.10+
- PyTorch
- Torchvision
- Scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenAI CLIP 

## Authors
- Akanksha Rai
- Zhanel Ashirbek
