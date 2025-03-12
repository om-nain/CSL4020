# **CNN-Based Autoencoder for SVHN Dataset using PyTorch**

## **1. Introduction**
In this project, we design and implement a **Convolutional Neural Network (CNN)-based Autoencoder** using **PyTorch** and train it on the **Street View House Numbers (SVHN) dataset**. The goal is to reconstruct images while applying regularization constraints to the model’s weights. 

We implement and compare two different autoencoders:
1. **Autoencoder with Weight Clipping**: Restricts all weights to the range **[-0.5, 0.5]**.
2. **Autoencoder with Weight Clipping and L1 Sparsity Regularization**: In addition to weight clipping, this model incorporates an **L1-based sparsity constraint**.

The models are trained under identical conditions, and their performance is evaluated using the **Peak Signal-to-Noise Ratio (PSNR)**. We also conduct **hyperparameter tuning**, experimenting with different activation functions and optimizers.

---

## **2. Dataset: Street View House Numbers (SVHN)**
### **2.1 Dataset Overview**
- The SVHN dataset consists of **real-world house number images** obtained from Google Street View.
- It contains **color images (32×32 pixels, RGB)**, making it a challenging dataset for autoencoder-based reconstruction.
- We use the dataset from `torchvision.datasets.SVHN`.

### **2.2 Data Preprocessing**
- Convert images to tensors and normalize them to the range **[-1, 1]**.
- Apply **random flipping and rotation** as data augmentation techniques.
- Convert images to **grayscale** (optional) for simpler encoding.
- Split the dataset into **training and validation sets**.

---

## **3. CNN-Based Autoencoder Architecture**
The autoencoder consists of an **encoder** and a **decoder**:

### **3.1 Encoder**
- **Convolutional layers** with **ReLU activation** to extract features.
- **Batch normalization** for stable training.
- **Max pooling layers** for downsampling.

### **3.2 Decoder**
- **Transpose convolution layers** to reconstruct the image.
- **Upsampling layers** to restore the original resolution.
- **Sigmoid activation** in the output layer to ensure pixel values are in the range `[0,1]`.

### **3.3 Regularization Constraints**
- **Weight Clipping**: All model weights are constrained to the range **[-0.5, 0.5]** after each training step.
- **L1 Sparsity Regularization**: A sparsity constraint is applied to encourage feature compression.

---

## **4. Training Strategy**
### **4.1 Loss Function**
- **Mean Squared Error (MSE) Loss** is used for image reconstruction.

### **4.2 Regularization Techniques**
- **Weight Clipping**: Ensures all model parameters remain within **[-0.5, 0.5]**.
- **L1 Regularization**: Encourages sparsity by penalizing large weight values.

### **4.3 Optimizers & Activation Functions**
We experiment with different:
- **Optimizers**: Adam, RMSprop, SGD
- **Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh

### **4.4 Training Conditions**
- Models are trained for **at least 100 epochs** or until the loss curve stabilizes.
- The learning rate is scheduled to decrease over time.

---

## **5. Model Evaluation**
### **5.1 Peak Signal-to-Noise Ratio (PSNR)**
- **PSNR** is used to measure the quality of reconstructed images.
- A **higher PSNR** indicates better reconstruction quality.

### **5.2 Loss Curve Analysis**
- Plot **training vs validation loss** to detect overfitting.
- Compare the loss curves of both models.

### **5.3 Visual Comparison of Reconstructed Images**
- Display sample reconstructed images from both autoencoders.
- Compare results qualitatively.

---

## **6. Hyperparameter Tuning**
- Experiment with **different optimizers and activation functions**.
- Evaluate **different learning rates**.
- Tune **dropout rates** and **batch sizes** to improve generalization.

---

## **7. Results & Analysis**
- Compare **PSNR scores** of both models.
- Discuss the impact of **L1 regularization** on sparsity.
- Evaluate the effectiveness of **weight clipping** in training stability.

---

## **8. Conclusion**
- Summarize key findings.
- Highlight challenges and future improvements.
- Discuss the trade-offs between **sparsity and reconstruction quality**.

---

## **9. References**
- PyTorch Documentation
- Research papers on CNN-based Autoencoders
- SVHN Dataset Information
