# **Transfer Learning Pipeline: Pre-training on STL-10 and Adapting to MNIST using PyTorch**

## **1. Introduction**
In this project, we implement a **transfer learning pipeline** using **PyTorch**. The approach involves:
1. **Pre-training a Convolutional Neural Network (CNN) on the STL-10 dataset**.
2. **Transferring the learned weights to the MNIST dataset**.
3. **Evaluating different fine-tuning strategies**:
   - Training only the linear (fully connected) layers.
   - Freezing initial layers and training the remaining feature extractor.
   - Fine-tuning the entire network.

We compare the performance of these strategies using **accuracy, per-class precision, recall, F1-score, and confusion matrices**.

---

## **2. Dataset Overview**
### **2.1 STL-10 Dataset (Source Domain)**
- **STL-10** is a dataset for **semi-supervised learning**, containing **color images (96x96 pixels, RGB)**.
- It has **10 classes** similar to CIFAR-10, with **5,000 training images** and **8,000 test images**.

### **2.2 MNIST Dataset (Target Domain)**
- **MNIST** consists of **grayscale images (28x28 pixels)** of handwritten digits **0-9**.
- It has **60,000 training images** and **10,000 test images**.

---

## **3. Data Preprocessing & Augmentation**
### **3.1 Preprocessing STL-10**
- Resize images to **96x96**.
- Normalize using mean and standard deviation.
- Convert images to tensors.
- Apply **data augmentation**:
  - Random cropping
  - Horizontal flipping
  - Gaussian blur

### **3.2 Preprocessing MNIST**
- Convert grayscale to **RGB** (3 channels) to match STL-10 format.
- Resize images to **96x96** to match STL-10.
- Normalize with mean and standard deviation.
- Convert to tensors.

---

## **4. Transfer Learning Pipeline**
### **4.1 Model Selection**
- Use a **CNN-based feature extractor**.
- Options:
  - Train a custom CNN from scratch on STL-10.
  - Use a **pre-trained model** (ResNet, VGG, MobileNet) trained on STL-10.

### **4.2 Training on Source Dataset (STL-10)**
- Train the CNN on STL-10 with:
  - **Cross-Entropy Loss**
  - **Adam optimizer**
  - **Learning rate scheduling**
- Log training **loss and accuracy**.

---

## **5. Transfer Learning Strategies**
After pre-training on STL-10, we adapt the model to MNIST using three approaches:

### **5.1 Training Only Linear Layers**
- **Freeze the entire feature extractor**.
- **Train only the final fully connected layers** on MNIST.

### **5.2 Freezing Initial Layers and Training Others**
- **Freeze the first few convolutional layers**.
- **Train deeper layers and fully connected layers**.

### **5.3 Fine-Tuning the Entire Network**
- **Unfreeze all layers**.
- Train the entire model on MNIST with a lower learning rate.

---

## **6. Model Evaluation**
### **6.1 Performance Metrics**
- **Overall Accuracy**
- **Per-Class Precision, Recall, and F1-Score**
- **Confusion Matrix**

### **6.2 Visualization of Predictions**
- Plot correct and misclassified MNIST images.
- Compare **performance across different transfer learning strategies**.

---

## **7. Computational Efficiency Analysis**
- Compare:
  - **Training time** for different fine-tuning strategies.
  - **Model parameter updates** in each approach.
  - **Memory usage** (GPU/CPU requirements).

---

## **8. Hyperparameter Tuning**
- Experiment with:
  - Different **learning rates** for fine-tuning.
  - **Batch sizes** for efficient training.
  - **Optimizers**: Adam, RMSprop, SGD.

---

## **9. Results & Discussion**
- Compare the effectiveness of:
  - Training only **linear layers** vs. partial feature extraction vs. full fine-tuning.
- Analyze **how much of the learned STL-10 knowledge transfers to MNIST**.
- Identify **best performing strategy** based on evaluation metrics.

---

## **10. Conclusion**
- Summarize key findings.
- Discuss **limitations and potential improvements**.
- Suggest **future research directions**.

---

## **11. References**
- STL-10 Dataset: [https://cs.stanford.edu/~acoates/stl10/](https://cs.stanford.edu/~acoates/stl10/)
- PyTorch Transfer Learning Guide
- Research papers on domain adaptation and knowledge transfer
