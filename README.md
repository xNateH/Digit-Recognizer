# Digit-Recognizer
This code provides a comprehensive overview of building, training, and evaluating a Multi-Layer Perceptron (MLP) model for image classification on the MNIST dataset using PyTorch. Below is a summary of the major steps in the code:

1. **Data Preparation:**
   - Downloads the MNIST dataset using the TensorFlow Keras library.
   - Splits the training set into a new training set and a validation set.
   - Displays information about the dataset, such as the number of samples, minimum and maximum intensity values, and image shapes.

2. **Data Loading:**
   - Converts the dataset into PyTorch tensors.
   - Creates DataLoader objects for batching the data.

3. **Model Architecture:**
   - Defines an MLP model with three fully connected layers.
   - Moves the model to the GPU if available.

4. **Model Summary:**
   - Prints the shape of the output tensor for each layer of the model.

5. **Optimizer:**
   - Defines an SGD optimizer with a specified learning rate for updating model parameters.

6. **Training Loop:**
   - Trains the MLP model on the training data for a specified number of epochs.
   - Computes and collects training and validation losses for each epoch.
   - Saves the model checkpoints at the end of each epoch.

7. **Learning Curve:**
   - Plots the learning curve showing training and validation losses for each epoch.

8. **Model Evaluation:**
   - Loads the weights of the best epoch based on validation loss.
   - Tests the model on the test set.
   - Plots the predicted probabilities for a sample image.
   - Computes and prints the overall accuracy of the model on the test set.
   - Generates and displays a confusion matrix to evaluate model performance.

9. **Final Thoughts:**
   - Provides insights into the model's performance and suggests potential improvements.

This code serves as a useful reference for building and training MLP models for image classification tasks using PyTorch and offers guidance on evaluating model performance.
