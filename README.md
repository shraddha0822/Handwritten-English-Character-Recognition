# Handwritten-English-Character-Recognition

This project aims to develop a Convolutional Neural Network (CNN) for recognizing handwritten alphabets. It utilizes image processing techniques, data preparation, and neural network design and training.

### Libraries Used

Numpy and Pandas: Data manipulation and analysis.

Matplotlib and OpenCV: Data visualization and image preprocessing.

Scikit-Learn: Dataset splitting and shuffling.

Keras and TensorFlow: CNN architecture and model training.


### Running the Notebook

To run this notebook, follow these steps:

1. **Open the Notebook in Google Colab**  
   Click [here](https://colab.research.google.com/github/shraddha0822/Handwritten-English-Character-Recognition/blob/main/Handwritten_English_Character_Recognition.ipynb) to open the notebook in Google Colab.

2. **Upload Your Kaggle API Key (kaggle.json)**  
   - Go to Kaggle's API section and create an API token if you haven’t already. This will download a `kaggle.json` file containing your Kaggle API key.
   - In Colab, upload the `kaggle.json` file by running the first code cell. Follow the instructions in the cell to securely configure your API key in Colab.

3. **Install Required Libraries**  
   The notebook installs all required libraries, so you don’t need any additional setup.

4. **Run All Cells**  
   - Once the setup is complete, select **Runtime > Run all** from the menu. This will run each cell sequentially.
   - The model will train on the AZ Handwritten Alphabets dataset, visualize data, and provide predictions.

5. **View Results**  
   After running the notebook, you can view the model summary, training accuracy, and sample predictions.


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
