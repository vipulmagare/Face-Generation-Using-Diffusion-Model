# Multi-Ethnic Face Generation using Diffusion Models

## Project Description
Engineered a sophisticated Diffusion Model using UNet2DModel architecture to generate high-quality synthetic face images across multiple ethnic categories (Indian, Asian, European), demonstrating advanced proficiency in generative AI and cross-cultural representation.

## 🚀 Overview
This project implements a state-of-the-art diffusion model that generates realistic human face images with ethnic diversity. The model uses a UNet2DModel architecture with DDPM (Denoising Diffusion Probabilistic Models) scheduler to create synthetic faces representing different ethnic backgrounds and their combinations.

## 🎯 Key Features
- **Multi-Ethnic Generation**: Creates faces from Indian, Asian (Oriental), and European categories
- **Hybrid Combinations**: Generates mixed-ethnicity faces (Oriental+Indian, Oriental+European, etc.)
- **Advanced Architecture**: Utilizes UNet2DModel for superior image generation quality
- **DDPM Scheduler**: Implements 1000-step denoising diffusion process
- **GPU Optimization**: Includes memory management for efficient training
- **Automated Dataset Processing**: Categorizes images based on filename prefixes

## 🏗️ Technical Architecture
- **Framework**: PyTorch with Diffusers library
- **Model**: UNet2DModel with 2 layers per block
- **Scheduler**: DDPMScheduler (1000 timesteps)
- **Image Size**: 64x64 pixels
- **Training**: AdamW optimizer with 5e-4 learning rate
- **Mixed Precision**: CUDA AMP support for memory efficiency

## 📁 Project Structure
```
face-diffusion-model/
├── diffusion_model.py          # Main training and generation script
├── generated_images/           # Output directory for synthetic faces
│   ├── OI/                    # Oriental+Indian combinations
│   ├── OE/                    # Oriental+European combinations  
│   ├── IE/                    # Indian+European combinations
│   └── OIE/                   # All three combinations
├── models/
│   └── diffusion_model.pth    # Trained model weights
├── dataset/                   # Training images (not included - see Dataset section)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔧 Requirements
```
torch>=1.13.0
torchvision>=0.14.0
diffusers>=0.21.0
transformers>=4.25.0
numpy>=1.21.0
Pillow>=9.0.0
google-colab (if running on Colab)
```

## 📊 Dataset Structure
The model expects a dataset with images named using specific prefixes:
- **Indian faces**: Files starting with "indian" (e.g., indian_001.jpg)
- **Asian faces**: Files starting with "asian" (e.g., asian_001.jpg) 
- **European faces**: Files starting with "white" (e.g., white_001.jpg)

### Dataset Setup
Due to GitHub's file size limitations, the dataset is hosted externally:

**Option 1: Download Dataset**
- Create a `dataset/` folder in your project directory
- Download your face dataset and organize with the naming convention above
- Update the `base_path` variable in the script

**Option 2: Use Your Own Dataset**
- Organize images in a single folder
- Rename files with appropriate prefixes (indian_, asian_, white_)
- Ensure images are in JPG/PNG format

## 🚀 Usage

### Training the Model
```python
# Update paths in the script
base_path = "path/to/your/dataset"
output_path = "path/to/output/directory"

# Run training
python diffusion_model.py
```

### Key Training Parameters
- **Epochs**: 1000 (adjustable)
- **Batch Size**: 8
- **Learning Rate**: 5e-4
- **Image Resolution**: 64x64
- **Timesteps**: 1000

### Generated Output
The model generates 10 images for each ethnic combination:
- **OI**: Oriental + Indian features
- **OE**: Oriental + European features  
- **IE**: Indian + European features
- **OIE**: All three ethnic combinations

## 📈 Model Performance
- **Architecture**: UNet2DModel with efficient 2-layer blocks
- **Training Stability**: MSE loss with AdamW optimization
- **Memory Efficiency**: GPU cache clearing and mixed precision
- **Generation Quality**: High-fidelity 64x64 synthetic faces
- **Diversity**: Multi-ethnic representation with hybrid combinations

## 🎨 Key Innovations
- ✅ **Ethnic Categorization**: Automated dataset organization by filename prefixes
- ✅ **Hybrid Generation**: Creates mixed-ethnicity synthetic faces
- ✅ **Memory Optimization**: Efficient GPU memory management
- ✅ **Robust Error Handling**: Comprehensive exception handling for dataset loading
- ✅ **Scalable Architecture**: Easy to extend to more ethnic categories

## 🔬 Technical Implementation Details

### Dataset Processing
```python
# Automatic categorization based on filename prefixes
category_map = {
    "indian": "Indians",
    "asian": "Orientals", 
    "white": "Europeans"
}
```

### Model Configuration
```python
model = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2
)
```

### Training Process
1. **Data Loading**: Custom dataset class with error handling
2. **Noise Addition**: Random noise injection at different timesteps
3. **Denoising**: Model learns to predict and remove noise
4. **Optimization**: AdamW with gradient-based learning
5. **Generation**: Reverse diffusion process for image synthesis

## 📋 Installation & Setup

### Option 1: Google Colab (Recommended)
```python
# Clone repository
!git clone https://github.com/yourusername/face-diffusion-model.git
%cd face-diffusion-model

# Install dependencies
!pip install diffusers transformers torch torchvision

# Mount Google Drive for dataset access
from google.colab import drive
drive.mount('/content/drive')
```

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/face-diffusion-model.git
cd face-diffusion-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Results & Applications
- **Research**: Cross-cultural AI representation studies
- **Art & Design**: Diverse character generation for media
- **Data Augmentation**: Synthetic face data for ML training
- **Bias Analysis**: Understanding model behavior across ethnicities
- **Creative Applications**: Art installations and digital media

## 🔮 Future Enhancements
- [ ] Higher resolution generation (128x128, 256x256)
- [ ] Conditional generation with age, gender, expression controls
- [ ] Additional ethnic categories and subcategories
- [ ] Real-time generation optimization
- [ ] Integration with face editing capabilities  
- [ ] Evaluation metrics for ethnic representation accuracy

## 📚 References & Inspiration
- **DDPM Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al.)
- **UNet Architecture**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Diffusers Library**: Hugging Face Diffusers documentation

## 🤝 Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.



## 👤 Contact
Feel free to reach out for questions, collaborations, or discussions about generative AI and ethical representation in machine learning!

---

**⭐ If you find this project useful, please give it a star!**

*This project demonstrates advanced skills in generative AI, deep learning, computer vision, and ethical AI development with focus on diverse representation.*
