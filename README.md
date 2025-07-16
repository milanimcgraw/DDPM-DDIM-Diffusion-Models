## 🤖 Building a Diffusion Model from Scratch

In this repo, we implement **Denoising Diffusion Probabilistic Models** and **Denoising Diffusion Implicit Models** from scratch, using the same generative AI techniques used in image generators like [DALL-E]( https://openai.com/dall-e) and [Stable Diffusion]( https://github.com/CompVis/stable-diffusion). 

**This project walks through a complete pipeline:** 
- implementing the mathematical foundations of both forward and reverse diffusion
- building and training neural networks (typically U-Net architectures) for noise prediction
- developing fast sampling algorithms
- extending the models with conditional generation features (classifier-free guidance and text/image-based prompts) for personalized image creation

## **Denoising Diffusion Probabilistic Models** (DDPMs) 
introduced by Ho et al. (2020) [arXiv:2006.11239](https://arxiv.org/abs/2006.11239), operate by gradually adding Gaussian noise when training images through a forward diffusion process, then learning to reverse the process using a neural network that predicts and removes noise at each timestep to generate high-quality images from pure noise. 

## **Denoising Diffusion Implicit Models** (DDIMs)
proposed by Song et al. (2020) [arXiv:2010.02502](https://arxiv.org/abs/2010.02502), accelerates generation by enabling deterministic, non-Markovian sampling that produces comparable results to DDPMs with significantly fewer denoising steps. 

## Project Overview 
1. Implement the basic diffusion algorithm.
2. Construct and train neural networks for noise prediction.
3. Explore sampling processes, both correct and incorrect.
4. Extending the model with contextual awareness for personalized image generation.
5. Add context for personalized image generation.

### ⚙️ Dependencies Installation:
- `torch` and `torchvision`: deep learning/computer vision tasks
- `tqdm`: progress bars
- `matplotlib`: plotting/visualization
- `numpy`: numerical operations
- `ipython`: interactive Python functionality
- `pillow`: image processing

### ⚙️ Imports:
- `from typing import Dict, Tuple`: type hinting capabilities
- `from tqdm import tqdm`: progress bar functionality
- `import torch`: PyTorch library
- `import matplotlib.pyplot as plt`: plotting capabilities
- `import numpy as np`: numpy for numerical operations
- `from IPython.display import HTML`: HTML display functionality (Jupyter notebooks)

### ⚙️ Diffusion Utilities:
- `class UnetUp(nn.Module)`: defines upsampling block for U-Net architecture, uses `ConvTranspose2d` for upsampling, `ResidualConvBlock` for feature processing
- `class UnetDown(nn.Module)`: defines downsampling block U-Net architecture, `ResidualConvBlock` and `MaxPool2d` for downsampling
- `class EmbedFC(nn.Module)`: implements feed-forward neural network for embedding, converts input data to a different dimensional space
- `def unorm(x)`: unity normalization function (scales data to [0,1] range)
- `def norm_all(store, n_t, n_s)`: applies unity norm to all timesteps, samples in input data
- `def norm_torch(x_all)`: applies unity norm to PyTorch tensor format data
- `def gen_tst_context(n_cfeat)`: generates test context vectors for experimentation
- `def plot_grid(x, n_sample, n_rows, save_dir, w)`: creates/saves grid of images for visualization
- `def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w, save=False)`: creates animated GIF of image evolution over time
- `class CustomDataset(Dataset)`: custom dataset class for handling sprite images and label, manages loading/preprocessing of image data

### ⚙️ Model Architecture:
- `class ContextUnet(nn.Module)`: implements U-Net architecture with added context awareness
  - Includes:
    - Initial convolution layer
    - Down-sampling path
    - Vector conversion
    - Context and time embedding
    - Up-sampling path
    - Final convolution layers
    - Forward method processes input image, timestep, and context

### ⚙️ Hyperparameters:
- `timesteps = 500`: sets number of timesteps for diffusion process
- `beta1 = 1e-4` and `beta2 = 0.02`: defines range of noise schedule
- `device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))`: sets device (GPU or CPU) for computation
- `n_feat = 64`, `n_cfeat = 5`, `height = 16`: define network parameters (like feature dimensions, image height)
- Noise schedule construction: creates noise schedule for diffusion process using defined beta values

### ⚙️ Additional Model Components:
- `class ResidualConvBlock(nn.Module)`: implements residual convolutional block for neural network, includes skip connections to help with gradient flow
- Redefined `class ContextUnet(nn.Module)`: provides detailed implementation of ContextUnet, includes precise layer definitions/connections, implements forward pass with context/time embedding

### ⚙️ Model Instantiation and Loading:
- `nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)`: creates instance of ContextUnet with specified parameters
- `nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth", map_location=device))`: loads pre-trained weights from a file
- `nn_model.eval()`: sets model to eval mode for inference

### ⚙️ Sampling Functions:
- `def denoise_add_noise(x, t, pred_noise, z=None)`: helper function for denoising process, removes predicted noise/adds controlled amount of new noise
- `def sample_ddpm(n_sample, save_rate=20)`: implements correct DDPM sampling algorithm, generates images: iteratively denoising from random noise
- `def sample_ddpm_incorrect(n_sample)`: demonstrates incorrect sampling w/o noise addition, shows importance of noise addition step in reverse process
- `def sample_ddim(n_sample, save_rate=20)`: implements DDIM sampling algorithm, provides faster sampling compared to DDPM by reducing the number of steps
- `def sample_ddim_context(n_sample, context, n=20)`: implements DDIM sampling with context

### ⚙️ Visualization:
- `samples, intermediate_ddpm = sample_ddpm(32)`: executes correct sampling process for 32 samples
- `animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=False)`: creates animation of correct sampling process
- `samples, intermediate = sample_ddpm_incorrect(32)`: executes incorrect sampling process for 32 samples
- `animation = plot_sample(intermediate, 32, 4, save_dir, "ani_run", None, save=False)`: creates animation of incorrect sampling process
- `samples, intermediate_ddim = sample_ddim(32)`: executes DDIM sampling process for 32 samples, demonstrating faster sampling
- `animation_ddim = plot_sample(intermediate_ddim, 32, 4, save_dir, "ani_run", None, save=False)`: creates animation of DDIM sampling process
- `samples, intermediate_ddim_context = sample_ddim_context(32, ctx)`: executes DDIM sampling with context for 32 samples
- `animation_ddpm_context = plot_sample(intermediate_ddim_context, 32, 4, save_dir, "ani_run", None, save=False)`: creates animation of DDIM sampling with context

### ⚙️ Display Animations:
- `HTML(animation_ddpm.to_jshtml())` and `HTML(animation.to_jshtml())`: displays the animations for DDPM and incorrect DDPM
- `HTML(animation_ddim.to_jshtml())`: displays the animation for DDIM
- `HTML(animation_ddpm_context.to_jshtml())`: displays the animation for DDIM with context

### ⚙️ Comparing DDPM and DDIM Speed:
- To compare the speed of DDPM and DDIM sampling algorithms, you can use the following code snippets:

```python
import time

# Compare DDPM and DDIM speed
n_samples = 32

# DDPM Sampling Time
start_time = time.time()
samples_ddpm, _ = sample_ddpm(n_samples)
ddpm_time = time.time() - start_time
print(f"DDPM sampling time: {ddpm_time:.2f} seconds")

# DDIM Sampling Time
start_time = time.time()
samples_ddim, _ = sample_ddim(n_samples)
ddim_time = time.time() - start_time
print(f"DDIM sampling time: {ddim_time:.2f} seconds")```

---
> ## 📌 Credits
> This code is modified from, [cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion). Diffusion model is based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). Type Markdown and LaTeX: 𝛼2.

