# **CNN Architecture with Depth-Wise Separable Convolutions for CIFAR-10**

## **1. Introduction**
In this project, we design and implement a **Convolutional Neural Network (CNN) architecture** utilizing **four dense blocks** with **depth-wise separable convolutions**. The model is trained on the **CIFAR-10 dataset**, and its performance is evaluated using **classification accuracy, precision, recall, and F1-score**.

Additionally, we analyze the computational efficiency and effectiveness of **depth-wise and point-wise convolutions** compared to **traditional convolutional layers**. The model is trained for **at least 100 epochs** or until the loss curve stabilizes.

---

## **2. CIFAR-10 Dataset Overview**
### **2.1 Dataset Characteristics**
- The **CIFAR-10 dataset** consists of **60,000 color images (32×32 pixels, RGB)**, classified into **10 categories**.
- There are **50,000 training images** and **10,000 test images**.

### **2.2 Data Preprocessing**
- Convert images to tensors and normalize them to the range **[-1, 1]**.
- Apply **data augmentation techniques**:
  - Random horizontal flipping
  - Random cropping
  - Normalization
- Split the dataset into **training, validation, and test sets**.

---

## **3. CNN Architecture with Depth-Wise Separable Convolutions**
### **3.1 Dense Blocks**
- The CNN consists of **four dense blocks**, each containing:
  - **Depth-wise separable convolutions** to improve computational efficiency.
  - **Batch normalization** for stabilizing training.
  - **ReLU activation** for non-linearity.
  - **Dropout** for regularization.

### **3.2 Depth-Wise Separable Convolutions**
- Instead of traditional convolutions, we use **depth-wise separable convolutions**, which involve:
  1. **Depth-wise Convolution**: Applies a single convolutional filter per input channel.
  2. **Point-wise Convolution**: Uses **1×1 convolutions** to combine the outputs.
- This reduces computational cost while maintaining model expressiveness.

### **3.3 Model Layers**
The model includes:
- **Input Layer**: Accepts 32×32×3 CIFAR-10 images.
- **Four Dense Blocks**: Each with depth-wise separable convolutions.
- **Global Average Pooling**: Reduces spatial dimensions before the output layer.
- **Fully Connected Layer (Softmax Activation)**: Outputs 10 class probabilities.

---

## **4. Training Strategy**
### **4.1 Loss Function**
- **Cross-Entropy Loss** is used for multi-class classification.

### **4.2 Optimizers**
- Experiment with different optimizers:
  - **Adam**
  - **SGD with momentum**
  - **RMSprop**

### **4.3 Training Conditions**
- Train for **at least 100 epochs** or until the loss curve stabilizes.
- Use **learning rate scheduling** to adjust the learning rate dynamically.

---

## **5. Model Evaluation**
### **5.1 Performance Metrics**
- **Classification Accuracy**: Measures overall prediction correctness.
- **Precision & Recall**: Evaluates false positives and false negatives.
- **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.

### **5.2 Computational Efficiency Analysis**
- Compare the computational efficiency of:
  - **Depth-wise separable convolutions** vs. **traditional convolutions**.
  - Measure **FLOPs (Floating Point Operations per Second)**.
  - Analyze **inference speed and memory consumption**.

### **5.3 Loss Curve Analysis**
- Compare **training vs validation loss**.
- Determine if the model is **overfitting or underfitting**.

---

## **6. Hyperparameter Tuning**
- Experiment with different:
  - Activation functions (**ReLU, Leaky ReLU, Swish**).
  - Optimizers (**Adam, RMSprop, SGD**).
  - Learning rates and batch sizes.
- Fine-tune dropout rates for optimal regularization.

---

## **7. Results & Discussion**
- Compare the accuracy and efficiency of:
  - **Depth-wise separable convolutions** vs. **Traditional convolutions**.
- Analyze the **trade-offs between computational cost and classification accuracy**.
- Evaluate whether **depth-wise separable convolutions** maintain high performance while reducing complexity.

---

## **8. Conclusion**
- Summarize the key takeaways from model evaluation.
- Discuss the **effectiveness of depth-wise separable convolutions** in CNN-based classification.
- Highlight potential **improvements and future research directions**.

---

## **9. References**
- PyTorch Documentation
- Research papers on depth-wise separable convolutions
- CIFAR-10 dataset description
