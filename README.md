# CNN_flower_recognition
Used Convolutional neural network for recognising different types of flowers

Dataset: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

 cross-entropy loss function. The model is trained for 50 epochs with a batch size of 128.

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
