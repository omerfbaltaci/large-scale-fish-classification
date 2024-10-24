# Fish Species Classification with Deep Learning

This project is a fish species classification task using a deep learning model. The dataset consists of various fish images from the Kaggle [Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset). The goal of this project is to classify fish images into their respective species using a convolutional neural network (CNN).

## Dataset
The dataset contains fish images in `.png` format and is organized into directories for each fish species. It has been split into training and validation sets for model training and evaluation.

## Project Structure
- **Data Preprocessing**: The images are rescaled and augmented to enhance model performance.
- **Model Architecture**: A simple CNN model with Dense and Dropout layers to prevent overfitting.
- **Model Evaluation**: Confusion matrix, classification report, and accuracy metrics were used to evaluate model performance.
- **Optimizer Comparison**: Different optimizers like Adam and RMSprop were tested to find the best performing optimizer.

## Dependencies
To run this project, the following Python libraries are required:
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies via:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fish-classification.git
    ```
2. Run the Jupyter notebook or Python script to train the model and evaluate the results.

## Model Summary
The model is built using the following layers:
- **Input Layer**: 150x150x3 (image size).
- **Flatten Layer**: Converts image data into 1D.
- **Dense Layers**: Three fully connected layers with ReLU activation.
- **Dropout Layer**: To reduce overfitting.
- **Output Layer**: Softmax activation for multi-class classification.

## Results
The model was trained using different optimizers and the results were evaluated based on accuracy and loss metrics. The final model achieved a good accuracy on the validation set, with a well-distributed confusion matrix.

### Visualizations
- **Training vs Validation Loss**: The loss plot shows the training and validation loss over each epoch.
- **Confusion Matrix**: Visualizes the model's performance on each fish species.

## Kaggle Notebook
https://www.kaggle.com/code/omerfbaltaci/large-scale-fish-classification

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```

Bu README.md dosyası, proje hakkında genel bir özet sunar, nasıl çalıştırılacağını açıklar ve sonuçları paylaşır. Kendi GitHub ve Kaggle linklerini ekleyerek kullanabilirsin.
