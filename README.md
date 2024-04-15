# Flower Classification with Deep Learning

This project showcases the development and application of a deep learning model for classifying images of flowers into five distinct categories: Daisy, Sunflower, Tulip, Dandelion, and Rose. The model is constructed using a Convolutional Neural Network (CNN) and is trained on a dataset of flower images. The project encompasses various stages, including data preprocessing, model architecture design, training, and evaluation.


Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Results](#visualizing-results)
- [Conclusion](#conclusion)
- [Key Functions](#key-functions)

## Introduction

The goal of this project is to build a deep learning model capable of accurately classifying images of flowers into one of the five categories. This is achieved through a series of steps, including data collection, preprocessing, model design, training, and evaluation.

## Requirements

To run this project, you will need Python 3.6 or higher and the following Python libraries:

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

The dataset for this project consists of images of flowers, each categorized into one of the five classes. The images are stored in separate directories for each flower type. The `make_train_data` function is used to read the images from these directories, resize them to a standard size (150x150 pixels), and append them to a list along with their corresponding labels.

## Model Architecture

The model architecture is designed to include a series of convolutional layers, each followed by a max pooling layer, and then a fully connected layer at the end. The model uses the Adam optimizer with a learning rate of 0.001 and the categorical cross-entropy loss function. The model is trained for 50 epochs with a batch size of 128.

## Training the Model

The model is trained using the `ImageDataGenerator` from Keras for data augmentation. This includes random rotations, zooms, shifts, and flips of the images. The training process also includes a callback to reduce the learning rate when the validation accuracy plateaus.

## Evaluating the Model

After training, the model's performance is evaluated on a test set. The loss and accuracy metrics are plotted over the epochs to visualize the training process. The model's predictions on the test set are compared with the actual labels to assess its performance.

## Visualizing Results

The project includes visualizations of the model's predictions on both correctly and incorrectly classified images. This helps in understanding the model's performance and identifying areas for improvement.

## Conclusion

This project demonstrates the process of building, training, and evaluating a deep learning model for image classification. The model achieves a reasonable level of accuracy on the flower dataset, showcasing the power of CNNs for image recognition tasks.

## Key Functions

- **`make_train_data(flower_type, DIR)`**: Reads images from a specified directory, resizes them, and appends them to a list along with their corresponding labels.
- **`assign_label(img, flower_type)`**: Assigns a label to an image based on the flower type.
- **`train_test_split(X, Y, test_size=0.25, random_state=42)`**: Splits the dataset into training and testing sets.
- **`ImageDataGenerator`**: Used for data augmentation during training.
- **`model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])`**: Compiles the model with the specified optimizer, loss function, and metrics.
- **`model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test), verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)`**: Trains the model for a specified number of epochs.
- **`model.evaluate(x_test, y_test)`**: Evaluates the model's performance on the test set.
- **`model.predict(x_test)`**: Makes predictions on the test set.

---

This README provides a detailed overview of the project, including the purpose, requirements, data preparation, model architecture, training process, evaluation, and visualization of results. It also highlights key functions used throughout the project, serving as a guide for anyone looking to understand or replicate the project.
