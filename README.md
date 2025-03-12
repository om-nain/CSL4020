# **CSL4020 - Deep Learning**
📍 **Indian Institute of Technology (IIT) Jodhpur**  
📆 **Spring 2025**  
👨‍🏫 **Instructor**: Dr. Angshuman Paul

---

## **📌 Course Overview**
**CSL4020 - Deep Learning** is a graduate-level course at **IIT Jodhpur**, covering fundamental and advanced topics in deep learning. The course involves:
- Neural networks and backpropagation
- Convolutional Neural Networks (CNNs)
- Autoencoders and representation learning
- Transfer learning and domain adaptation
- Semantic segmentation
- Hyperparameter tuning and optimization
- Applications in computer vision and natural language processing

This repository contains **assignments, projects, and implementation notebooks** for the course.

---

## **📂 Course Assignments**
### ✅ **Completed**
| **Assignment**  | **Topics Covered** | **Dataset** |
|---------------|-------------------|------------|
| **MLP for MNIST Classification** | Multi-layer perceptron (MLP), dropout, batch normalization | MNIST |
| **CNN-Based Autoencoder** | Convolutional autoencoder, weight clipping, L1 regularization | SVHN |
| **Depth-wise Separable CNN** | Efficient CNNs, depth-wise separable convolutions | CIFAR-10 |

### ⏳ **Ongoing**
| **Assignment**  | **Topics Covered** | **Dataset** |
|---------------|-------------------|------------|
| **SegNet for Semantic Segmentation** | Fully convolutional networks, encoder-decoder architectures | Pascal VOC |
| **Transfer Learning: STL-10 → MNIST** | Domain adaptation, feature extraction, fine-tuning | STL-10, MNIST |

---

## **⚙️ Installation**
### **Prerequisites**
Ensure you have **Python 3.7+** and the necessary dependencies installed.

```bash
# Clone the repository
git clone https://github.com/om-nain/CSL4020.git
cd CSL4020

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
