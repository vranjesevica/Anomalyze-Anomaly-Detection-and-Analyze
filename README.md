# Anomalyze - Anomaly Detection and Analyze

## About the Project
This project focuses on anomaly detection for industrial applications using the **MVTec AD dataset**. The primary goal is to identify defective regions in images and classify them as anomalies. The approach utilizes a deep learning autoencoder architecture to reconstruct normal data and detect deviations.  

Key technologies and techniques used:
* PyTorch
* Autoencoder
* Residual Network
* ImageNet
* Anomalib

### Motivation for the Project
The motivation for this project stemmed from my university professor after I expressed a desire to learn more about deep learning techniques for image analysis. The project guided me through the independent creation of an autoencoder network, including:

-   Designing an encoder to identify and extract key features.
-   Constructing a latent space and exploring its dimensionality.
-   Reconstructing images based on features extracted from the latent space.

After completing the foundational part, I utilized the MVTec dataset and incorporated pre-trained models, such as ImageNet, for fine-tuning and adapting them to my dataset.

During the project, I explored various techniques, including:

-   Transfer learning
-   Fine-tuning
-   One-class classification
-   Image classification
-   Data augmentation
-   Visualization techniques
-   Reconstruction error analysis

This project has taught me about building autoencoders, different types of convolutional layers and networks, leveraging pre-trained models, and fine-tuning models for specialized applications

### Organisation of the project
In Version 1 there is folders with different categories of dataset. Each of them has pth file, which represents saved trained model, and ipynb file or files with code and results.
In Version 2 architecture of the project is different because I use pretrained model so there are two ipynb files in which all the code and results can be found.

## Dataset
The MVTec AD dataset is specifically designed for anomaly detection tasks. It contains 15 categories of objects. Each category includes images divided into training, testing, and ground truth masks. Below is an example from the official dataset:
---------------------------------------------------------- SLIKA DATASETA ----------------------------------------------------
## Model Architecture
This project was approached as a research and learning experience, so multiple solutions were explored for the problem, resulting in various models that performed well.

### Version 1
The primary goal was to understand how autoencoders function, which led me to implement a custom autoencoder. Firstly, I implemented Residual Block which was adaptive for different number of in and out channels and modes so it was used to build Residual Network for encoder and inverted Residual Network for decoder.

The encoder is implemented as a custom PyTorch module (`ResNetEncoder`) that extracts hierarchical features from input images and maps them to a latent space. The architecture consists of three main components:

-   **Input Network**: A single `Conv2d` layer with ReLU activation.
-   **Body Network**: A sequence of `ResidualBlock` modules arranged hierarchically. The network progressively downsamples the input using residual blocks with "downsample" modes, increasing feature dimensionality to 256 channels.
-   **Output Network**: Applies `AdaptiveAvgPool2d` to reduce the feature map to a fixed spatial size. Features are flattened and passed through a fully connected layer to produce the latent vector of a specified dimension (`latent_dim`).
This version of the project is located in Models/Models. All of this was firstly implemented on Cifar dataset, and later trained and tested on some categories from MVTec Dataset. There is also differences in resizing images which led to different results in the end, and all of them can be found on the same path.

For the best results of version 1 try looking at Hazelnut part of the dataset. On that dataset, my models works excellent, especially Hazelnut128.ipynb.

### Version 2
After testing my autoencoder, I found that while it performed well, its quality depended on the data category and the image resizing during preprocessing. While it worked effectively, I wanted to explore more advanced solutions.
I researched pre-trained models and selected an **autoencoder trained on ImageNet**. I fine-tuned these architectures on my dataset and analyzed their results.
Additionally, I utilized the **Anomalib** library, an excellent tool for anomaly detection, which allowed me to explore its pre-implemented anomaly detection models.

## Installation
The dataset I am using is available on the official MVTec AD Dataset site and can be downloaded from there. The link to the site is: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad).

### Prerequisites
Ensure you have Python 3.7 or higher installed on your system.

### Steps

1.  **Clone the Repository**
    `git clone https://github.com/yourusername/yourproject.git`
    `cd yourproject` 
    
2.  **Set Up a Virtual Environment** (recommended)
    `python -m venv env`
    `source env/bin/activate` (On Windows, use `env\Scripts\activate`)
    
3.  **Install Required Libraries**  
    Install the necessary Python packages using `pip` by running:
    `pip install torch torchvision pytorch-lightning tqdm matplotlib pillow numpy` 

4. **Enable MPS on macOS**  
If you are using a Mac with an Apple Silicon chip (M1, M2, etc.), PyTorch supports **MPS** for accelerated training and inference. To enable MPS:

-   Ensure you have PyTorch 1.12 or newer installed.
-   Verify MPS support by running:
    `python -c "import torch; print(torch.backends.mps.is_available())"` 
    
-   If MPS is available, the project automatically detects and uses it during training.    
5.  **Additional Dependencies**  
    If the project includes custom modules (e.g., `vgg`, `resnet`), ensure they are in the correct folder structure.
    
    For example, if these are part of your repository, they should be in the same directory as your main script or within a package structure. Update `PYTHONPATH` if necessary:
    `export PYTHONPATH=$PYTHONPATH:/path/to/your/project` 
    
6.  **Verify Installation**  
    Test the setup by running a script or checking library imports:    
    `python -c "import torch; print(torch.__version__)"` 
#### Note:
-   If running on macOS, the project supports MPS acceleration. You can set the device explicitly in the code:
    `device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")` 
-   For other platforms, ensure you have **GPU drivers and CUDA Toolkit** installed if using GPU acceleration.
-   This project uses **PyTorch Lightning** for model training and checkpointing.

## Usage


## Results

### Version 1 Results
As we can see my Autoencoder is able to reconstruct images in the way that separetes anomalies and is able to mark them. 

<div>
    <img src='https://github.com/user-attachments/assets/6a6e7928-ba9c-4ad4-8294-3b6fdf17d527'>
    <img src='https://github.com/user-attachments/assets/2324f24e-3a3e-4b26-865a-e29dde7e9858'>
</div>



### Version 2 Results
On the images below it is shown that model recognises on which images there is anomaly detected and on which images there is no anomaly.

<div>
    <img src='https://github.com/user-attachments/assets/c21fe4a0-17ff-49af-8e23-75dd3fafad10'>
    <img src='https://github.com/user-attachments/assets/c6809978-2835-4c26-97e2-d4fece172441'>
    <img src='https://github.com/user-attachments/assets/f5361d30-52df-403b-9df9-18798c40d98e'>
    <img src='https://github.com/user-attachments/assets/1cbae527-884e-4bc3-be36-0688003e2ad4'>
</div>


