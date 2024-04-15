# Flower Classification with Deep Learning

This project demonstrates how to build a deep learning model for classifying images of flowers into five categories: Daisy, Sunflower, Tulip, Dandelion, and Rose. The model is built using a Convolutional Neural Network (CNN) and is trained on a dataset of flower images. The project includes data preprocessing, model training, and evaluation steps.
The model is trained for 50 epochs with a batch size of 128.

Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition



## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Results](#visualizing-results)
- [Conclusion](#conclusion)

## Requirements

This project requires Python 3.6 or higher and the following libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (cv2)
- TensorFlow
- Keras
- Scikit-learn
- PIL

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras scikit-learn pillow
```

## Data Preparation

The dataset consists of images of flowers, each categorized into one of the five classes. The images are stored in separate directories for each flower type. The `make_train_data` function reads the images from these directories, resizes them to a standard size (150x150 pixels), and appends them to a list along with their corresponding labels.

## Model Architecture

The model architecture consists of a series of convolutional layers, each followed by a max pooling layer, and then a fully connected layer at the end. The model uses the Adam optimizer with a learning rate of 0.001 and the categorical cross-entropy loss function. The model is trained for 50 epochs with a batch size of 128.

## Training the Model

The model is trained using the `ImageDataGenerator` from Keras for data augmentation. This includes random rotations, zooms, shifts, and flips of the images. The training process also includes a callback to reduce the learning rate when the validation accuracy plateaus.

## Evaluating the Model

After training, the model's performance is evaluated on a test set. The loss and accuracy metrics are plotted over the epochs to visualize the training process. The model's predictions on the test set are compared with the actual labels to assess its performance.

## Visualizing Results

The project includes visualizations of the model's predictions on both correctly and incorrectly classified images. This helps in understanding the model's performance and identifying areas for improvement.

## Conclusion

This project demonstrates the process of building, training, and evaluating a deep learning model for image classification. The model achieves a reasonable level of accuracy on the flower dataset, showcasing the power of CNNs for image recognition tasks.

---

This README provides a comprehensive overview of the project, including the purpose, requirements, data preparation, model architecture, training process, evaluation, and visualization of results. It serves as a guide for anyone looking to understand or replicate the project.
