# Handwritten-English-Character-Recognition

### Run the Notebook in Google Colab

[Open in Colab](https://colab.research.google.com/github/shraddha0822/Handwritten-English-Character-Recognition/blob/main/Handwritten_English_Character_Recognition.ipynb)


This project aims to develop a Convolutional Neural Network (CNN) for recognizing handwritten alphabets. It utilizes image processing techniques, data preparation, and neural network design and training.

### Libraries Used

Numpy and Pandas: Data manipulation and analysis.

Matplotlib and OpenCV: Data visualization and image preprocessing.

Scikit-Learn: Dataset splitting and shuffling.

Keras and TensorFlow: CNN architecture and model training.

### Link to dataset :
 https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

### CNN Architecture

1. Layers:

Conv2D and MaxPooling layers for feature extraction.

Dense layers for classification.

Softmax output layer with 26 units (A-Z) for classification.



2. Compiling:

Optimizer: Adam

Loss Function: categorical_crossentropy

Metric: accuracy



3. Training:

Model is trained on the training set with validation on the test set.

### Evaluation

Training Accuracy: ~95.67%

Validation Accuracy: ~97.99%

Training Loss: ~0.156

Validation Loss: ~0.071


### Prediction and Visualization

Random samples from the test set are visualized along with predicted labels to demonstrate the model's accuracy in recognizing handwritten letters.
