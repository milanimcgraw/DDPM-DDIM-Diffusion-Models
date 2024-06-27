# 🤖 Building a Diffusion Model from Scratch

In this repo, we'll be creating a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch, sampling & training diffusion models, building neural networks for noise prediction, & adding context for personalized image generation. This repo serves as both an educational resource and a practical guide to building and understanding one of the most powerful GenAI models used today. 

**Here,** you will find the overview code components, images of completed code (with outputs--see Denoising-Diffusion-Probabilistic-Model-DDPM-from-Scratch/diffusion model images), and a walkthrough of some of the code. 

## ⚙️The Build Process (Deliverables)
* Implement the basic diffusion algorithm
* Construct and train neural networks for noise prediction
* Explore sampling processes, both correct and incorrect
* Extending the model with contextual awareness for personalized image generation
* Add context for personalized image generation

## Overview 🏗️

### ⚙️ Dependencies Installation:
- **torch and torchvision:** deep learning/computer vision tasks
- **tqdm:** progress bars
- **matplotlib:** plotting/visualization
- **numpy:** numerical operations
- **ipython:** interactive Python functionality
- **pillow:** image processing

### 🛠️ Imports:
- **from typing import Dict, Tuple:** type hinting capabilities
- **from tqdm import tqdm:** progress bar functionality
- **import torch:** PyTorch library 
- **import matplotlib.pyplot as plt:** plotting capabilities 
- **import numpy as np:** numpy for numerical operations
- **from IPython.display import HTML:** HTML display functionality (Jupyter notebooks)

### 🛠️ Diffusion Utilities:
- **class UnetUp(nn.Module):** defines upsampling block for U-Net architecture, uses ConvTranspose2d for upsampling, ResidualConvBlock for feature processing
  
- **class UnetDown(nn.Module):** defines downsampling block U-Net architecture, ResidualConvBlock and MaxPool2d for downsampling
  
- **class EmbedFC(nn.Module):** implements feed-forward neural network for embedding, converts input data to a different dimensional space

- **def unorm(x):** unity normalization function (scales data to [0,1] range)

- **def norm_all(store, n_t, n_s):** applies unity norm to all timesteps, samples in input data

- **def norm_torch(x_all):** applies unity norm to PyTorch tensor format data

- **def gen_tst_context(n_cfeat):** generates test context vectors for experimentation

- **def plot_grid(x, n_sample, n_rows, save_dir, w):** creates/saves grid of images for visualization

- **def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False):** creates animated GIF of image evolution over time

- **class CustomDataset(Dataset):** custom dataset class for handling sprite images and label, manages loading/preprocessing of image data

### 🛠️ Model Architecture:
**class ContextUnet(nn.Module):** implements U-Net architecture with added context awareness
### Includes:
	•	Initial convolution layer
	•	Down-sampling path
	•	Vector conversion
	•	Context and time embedding
	•	Up-sampling path
	•	Final convolution layers
	•	Forward method processes input image, timestep, and context

### 🛠️ Hyperparameters:
- **timesteps = 500:** sets number of timesteps for diffusion process
- **beta1 = 1e-4 and beta2 = 0.02:** defines range of noise schedule
- **device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu')):** sets device (GPU or CPU) for computation
- **n_feat = 64, n_cfeat = 5, height = 16:** define network parameters (like feature dimensions, image height)
- **Noise schedule construction:** creates noise schedule for diffusion process using defined beta values

### 🛠️ Additional Model Components:
- **class ResidualConvBlock(nn.Module):** implements residual convolutional block for neural network, includes skip connections to help with gradient flow

- **Redefined class ContextUnet(nn.Module):** provides detailed implementation of ContextUnet, includes precise layer definitions/connections, implements forward pass with context/time embedding

### 🛠️ Model Instantiation and Loading:
- **nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device):** creates instance of ContextUnet with specified parameters

- **nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth", map_location=device)):** loads pre-trained weights from a file

- **nn_model.eval():** sets model to eval mode for inference

### 🛠️ Sampling Functions:
- **def denoise_add_noise(x, t, pred_noise, z=None):** helper function for denoising process, removes predicted noise/adds controlled amount of new noise

- **def sample_ddpm(n_sample, save_rate=20):** implements correct DDPM sampling algorithm, generates images: iteratively denoising from random noise

- **def sample_ddpm_incorrect(n_sample):** demonstrates incorrect sampling w/o noise addition, shows importance of noise addition step in reverse process

### 🔍 Visualization:
- **samples, intermediate_ddpm = sample_ddpm(32):** executes correct sampling process for 32 samples
  
- **animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=False):** creates animation of correct sampling process

- **samples, intermediate = sample_ddpm_incorrect(32):** executes incorrect sampling process for 32 samples
  
- **animation = plot_sample(intermediate, 32, 4, save_dir, "ani_run", None, save=False):** creates animation of incorrect sampling process

- **HTML(animation_ddpm.to_jshtml()) and HTML(animation.to_jshtml()):** displays the animations

> [!NOTE]
> This code is modified from: https://github.com/cloneofsimo/minDiffusion.
> Diffusion model is based on Denoising.
> Diffusion Probabilistic Models and Denoising Diffusion Implicit Models.
> Type Markdown and LaTeX: 𝛼2.


