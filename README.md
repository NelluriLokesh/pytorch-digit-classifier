# Handwritten Digit Recognition using Autoencoders (MNIST)

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## 📌 Project Overview
This project builds an **Autoencoder** using **PyTorch** to learn a compressed representation of handwritten digit images from the **MNIST dataset**. The model is trained to **encode and reconstruct** the images, demonstrating feature extraction capabilities.

## 🚀 Technologies Used
- Python  
- PyTorch  
- torchvision  
- NumPy  
- Matplotlib  

## 📂 Dataset
The project uses the **MNIST dataset**, which contains **70,000 grayscale images** of handwritten digits (0-9).
- **Training Set**: 60,000 images  
- **Test Set**: 10,000 images  

## 🏗 Project Structure
```
📂 Handwritten-Digit-Recognition-Autoencoder/
│── Project on Autoencoders using Handwritten Digit dataset.ipynb  # Jupyter Notebook
│── dataset/                                                       # (Optional) Dataset Folder
│── models/                                                        # Trained models (if any)
│── README.md                                                      # Project Documentation
```

## 📖 How It Works
1. **Preprocess Data**  
   - Load MNIST dataset using `torchvision.datasets`  
   - Normalize and transform images into tensors  
2. **Define Autoencoder Model**  
   - Encoder: Compresses input images into a lower-dimensional representation  
   - Decoder: Reconstructs the original image from encoded features  
3. **Train the Model**  
   - Uses **Mean Squared Error (MSE) Loss**  
   - Optimizer: **Adam**  
   - Trained over multiple epochs  
4. **Evaluate and Visualize Results**  
   - Compare **original vs. reconstructed** images  
   - Plot training loss over epochs  

## 🔧 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/NelluriLokesh/Handwritten-Digit-Recognition-Autoencoder.git
cd Handwritten-Digit-Recognition-Autoencoder
```
### 2️⃣ Install Dependencies
```bash
pip install torch torchvision numpy matplotlib
```
### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Open **"Project on Autoencoders using Handwritten Digit dataset.ipynb"** and execute the cells.

## 📊 Sample Output
🔹 **Original vs. Reconstructed Images**
```python
import matplotlib.pyplot as plt
# Sample code to visualize results
```
![Sample Output](https://raw.githubusercontent.com/github/explore/main/topics/mnist/mnist.png)

## 🏆 Results & Insights
✅ Successfully trained an autoencoder to compress and reconstruct MNIST images.  
✅ Achieved good reconstruction quality with minimal loss.  
✅ The latent space can be used for **dimensionality reduction** or **anomaly detection**.  

## 📌 Future Enhancements
🔹 Improve reconstruction quality with **deeper networks**.  
🔹 Experiment with **variational autoencoders (VAEs)**.  
🔹 Apply to **denoising applications**.  

## 📜 License
This project is open-source and available under the **MIT License**.
