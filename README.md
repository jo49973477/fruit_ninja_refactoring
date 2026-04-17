# FruitNInja Refactoring - The Simpler FruitNinja!

Welcome! This repository is a lightweight, refactored version of the CVPR 2025 paper, [FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](https://arxiv.org/abs/2411.12089).
While the original codebase is tightly integrated with PhysGaussian, this project aims to isolate FruitNinja's core features. By removing unused dependencies and refactoring the pipeline, this repo resolves potential "dependency hell" and offers a cleaner, ready-to-use environment.
Feel free to use this streamlined codebase for your own research. I am always open to feedback!

## 1. How to install Dependency
At eleast CUDA 12.1 is supported
If you use ```uv```, you can just simply use
```bash
uv sync
```
If you want to use ```pip```, no worries! It is way simple, too!
```bash
pip install .
```

## 2. Leveraging Hydra

This script leverages ```hydra``` to manage configurations efficiently. You don't need to pass dozens of arguments through the command line; simply configure your ```.yaml``` file or override specific parameters on the fly.

## 3. Dreambooth Fine-Tuning


### Step 1: Prepare Your Data
Ensure your instance images (e.g., cross-sections) are located in the directory specified by ```instance_data_dir```. The script will automatically generate the corresponding depth maps using MiDaS, so you don't need to generate them manually!

- **Form A:** Inside ```instance_data_dir```, you can save your Dataset in two ways. Here **form A** is suggested:

```
 ㄴ vertical
    ㄴ image1.jpg
    ㄴ image2.jpg
    ㄴ ...
 ㄴ horizontal
    ㄴ image1.jpg
    ㄴ image2.jpg
    ㄴ ...
 ㄴ vanilla
    ㄴ image1.jpg
    ㄴ image2.jpg
    ㄴ ...
```
Inside the ```vertical```, ```horizontal```, ```vanilla```, you have to put images which represent vertical-cross section, horizontal-cross section, filled images.

- **Form B:** If you want to give all other prompts per each image, you should use **Form B** all images must have prompt in json file.
Json file should have a dictionary, whose key is name of image file and value is prompt
```
name_of_image : prompt
```

```
 ㄴ image1.jpg
 ㄴ image2.jpg
 ㄴ ...
 ㄴ metadata.json
```
Like such, json file must be found!


### Step 2: Run the Training Script
Since the project uses Hydra, you can just add your ```yaml``` configuration file in the directory ```./config``` and start the training by running:

```bash
python dreambooth.py --config-name=your_conf
```

### Step 3: Override Parameters on the Fly (Optional)
If you want to test different parameters without editing the config file, use Hydra's command-line overrides:

```bash
python dreambooth.py --config-name=your_conf learning_rate=1e-5 train_batch_size=2 use_8bit_adam=True
```

### ```FinetuneConfig``` Parameter Breakdown
Here is a breakdown of the essential parameters you need to care about:

- ```pretrained_model_name_or_path```: The base Stable Diffusion Depth2Img model (e.g., "sd2-community/stable-diffusion-2-depth").
- ```pretrained_txt2img_model_name_or_path```: The base Text-to-Image model used exclusively for generating class images if prior preservation is enabled.
- ```instance_data_dir```: The folder path where your target images are stored.
- ```class_data_dir```: The folder path to store generated class images (only used if with_prior_preservation=True).
- ```prompt_json_dir```: Path to a JSON file containing specific prompts for each image. If not provided, the script falls back to folder-name-based prompting.
- ```nickname``` & ```class_prompt```: Used to construct the training prompt (e.g., "A [nickname] [class_prompt]").
- ```num_class_images```: The total number of class images to generate for prior preservation.
- ```resolution```: The size to which images will be resized and cropped (default: ```512```).
- ```output_dir```: The directory where the final LoRA weights or checkpoints will be saved.
- ```train_batch_size```: Number of images processed per batch per GPU.
- ```gradient_accumulation_steps```: Number of steps to accumulate gradients before updating weights. Useful for saving VRAM.
- ```mixed_precision```: Set to "fp16" or "bf16" to drastically reduce memory usage and speed up training.
- ```use_8bit_adam```: Set to True to use bitsandbytes 8-bit Adam optimizer (huge VRAM saver!).
- ```learning_rate```: The initial learning rate.
- ```with_prior_preservation```: Set to True to prevent the model from forgetting how to draw standard objects of the same class (reduces language drift).



## 4. Filling the fruit!

This script is designed to fill the empty internal spaces of your 3D Gaussian Splatting (3DGS) models using Taichi for GPU-accelerated computing.

### Step 1: Execute the Script
Since Hydra is hardcoded to look for `config_name="filling_config"`, you can start the process by simply running:
```bash
python inside_filling.py --config-name=your_conf
```

### Step 2: What Happens Under the Hood?
* **Taichi Initialization:** The script automatically fires up the Taichi CUDA engine and allocates 16.0GB of device memory to handle massive calculations efficiently.
* **Model Processing:** It loads your `.ply` file (either a standard point cloud or a 3DGS-formatted file), rotates it, and centers it.
* **Internal Filling:** Taichi calculates the internal volume based on your configured grid density and injects new points into the empty spaces.
* **Export:** The filled points are combined with the original surface points, recolored, and saved as a new `.ply` file that is ready for rendering.

### `FillingConfig` Parameter Breakdown

Here is a streamlined breakdown of the essential parameters for the internal filling process.

* **`model_path`**: The absolute or relative path to your input `.ply` file (the hollow 3D Gaussian Splatting model you want to fill).
* **`output_path`**: The destination path where the newly filled `.ply` file will be saved.
* **`white_br`**: A boolean flag (`True`/`False`) that defines the background color during processing. Set to `True` for a white background (represented as `[1, 1, 1]`) or `False` for a black background (`[0, 0, 0]`). This helps the model differentiate between the object and the empty space.
* **`particle_params`**: This sub-configuration dictates how Taichi injects particles into the empty internal volume. It controls parameters like grid density (`n_grid`), density thresholds, and ray-casting directions to ensure the inside is filled solidly without leaking outside the mesh boundaries.
* **`material_params`**: This defines the physical properties (like Poisson's ratio `nu`, and Young's modulus `E`) assigned to the newly generated internal points. While primarily used for downstream physics simulations, it ensures the internal structure behaves like a consistent solid (e.g., "jelly").

## 4. Train the FruitNinja!

This script is the core engine of your pipeline, designed to optimize the internal textures of your 3D Gaussian Splatting (3DGS) model using Score Distillation Sampling (SDS) guided by Stable Diffusion.

### Step 1: Run the Training Script
Ensure your filled `.ply` model (from `inside_filling.py`) is ready, and execute the following command:
```bash
python train.py --config-name=your_conf
```

### Step 2: Dynamic Overrides (Optional)

If you want to tweak specific training parameters without modifying the `.yaml` file, you can override them directly via the command line:
```bash
python train.py --config-name=your_conf epochs=500 guidance_scale=20 sds_steps=10
```

---

### `TrainerConfig` Parameter Breakdown

Here is a breakdown of the essential parameters you need to care about to control the training dynamics:

* **`output_path`**: The root directory where your rendered images, cached references, and final `.pt` models will be saved.
* **`white_bg`**: Determines the background color for rasterization (not actively used in the current script as `self.background` is hardcoded to white, but good for future-proofing).
* **`gaussian_path`**: The path to your filled, hollow-free `.ply` model (the output from the filling script) that you want to optimize.
* **`gaussian_orig`**: The path to the original, untouched 3DGS model. Used as a ground-truth reference to prevent the exterior from degrading during internal optimization.
* **`center_pos`**: The exact `[X, Y, Z]` coordinate representing the physical center of your object. Crucial for calculating correct camera angles and slicing masks.
* **`epochs`**: Total number of training loops.
* **`init_radius`**: The distance from the object center to the camera when rendering cross-sectional views.
* **`image_size`**: The resolution of the rendered cross-sections and diffusion outputs (default is 512).
* **`sds_per_epoch`**: Determines how often (every N epochs) the Stable Diffusion model generates a new reference image.
* **`sds_steps`**: The number of diffusion denoising steps applied during the SDS process to update the latent vector.
* **`guidance_scale`**: The Classifier-Free Guidance (CFG) scale. A higher value forces the generated texture to adhere more strictly to your text prompt.
* **`lrs`**: A dictionary containing the learning rates for different Gaussian attributes (means, scales, quats, opacities, colors) and the SDS latent vector.
* **`sd_model_vertical` & `sd_model_horizontal`**: The Hugging Face model IDs for the Depth-to-Image diffusion models used for different slicing directions.
* **`vertical_prompt` & `horizontal_prompt`**: The text prompts describing what the internal texture should look like (e.g., "macro photo of an orange cross-section").
* **`lambda_opaque`, `lambda_scale`, `lambda_iso`**: Weighting factors for the Opaque Atom regularization loss, designed to keep Gaussians small, opaque, and uniformly sized.
* **`opaque_atom`**: A boolean flag (`True`/`False`) to enable or disable the Opaque Atom regularization.
