# **MLP for MNIST Classification using PyTorch**

## **1. Introduction**
In this assignment, we implement a **Multi-Layer Perceptron (MLP)** using **PyTorch** to classify handwritten digits from the **MNIST dataset**. The dataset is preprocessed by reshaping each image into a **1D vector** and applying various transformations such as rotation and scaling. The model includes **dropout** and **batch normalization** layers to enhance generalization. Additionally, we employ **custom weight initialization** and analyze the impact of regularization techniques on overfitting.

---

## **2. Dataset Preprocessing**
### **2.1 Load the MNIST Dataset**
- The dataset is loaded using `torchvision.datasets.MNIST`.
- Each image is a **28x28 grayscale** image.
- Convert the images into **1D vectors** of **size 784**.

### **2.2 Data Augmentation & Transformation**
- Apply transformations such as:
  - **Random Rotation**
  - **Random Scaling**
  - **Normalization** (Mean = 0.5, Std = 0.5)
- Convert images to tensors and normalize them.
- Flatten each image from `28x28` into a `1D vector` of size `784`.

---

## **3. Building the MLP Model**
### **3.1 MLP Architecture**
The MLP consists of:
- **Input Layer**: 784 neurons (flattened 28x28 images).
- **Hidden Layers**: Multiple fully connected layers with ReLU activation.
- **Dropout**: Helps prevent overfitting.
- **Batch Normalization**: Improves training speed and stability.
- **Output Layer**: 10 neurons (digits 0-9) with Softmax activation.

### **3.2 Custom Weight Initialization**
- Define a custom function to initialize weights.
- Use techniques like **Xavier Initialization** or **He Initialization**.

---

## **4. Training the Model**
### **4.1 Define Loss Function and Optimizer**
- Use **CrossEntropy Loss** for multi-class classification.
- Use optimizers such as **Adam** or **SGD with momentum**.

### **4.2 Training Loop**
- Train the model for a set number of epochs.
- Perform forward and backward propagation.
- Apply dropout and batch normalization during training.

---

## **5. Model Evaluation**
### **5.1 Performance Metrics**
- Calculate **accuracy, loss**, and other evaluation metrics.
- Use the test dataset for evaluation.

### **5.2 Precision, Recall, and F1-Score**
- Compute **precision, recall, and F1-score** for each class.
- Compare model performance across different digits.

### **5.3 Confusion Matrix**
- Generate a **confusion matrix** to visualize classification performance.
- Analyze misclassified examples.

---

## **6. Regularization and Overfitting Analysis**
- Compare performance **with and without dropout**.
- Analyze how **batch normalization** affects model generalization.
- Observe **training vs. validation loss** to detect overfitting.

---

## **7. Hyperparameter Tuning**
- Experiment with different:
  - Learning rates
  - Optimizers
  - Dropout rates
  - Number of hidden layers
- Find the best combination that maximizes accuracy and minimizes overfitting.

---

## **8. Conclusion**
- Summarize the findings from the training and evaluation process.
- Discuss the impact of **regularization techniques** on model performance.
- Highlight key **challenges and improvements**.

---


