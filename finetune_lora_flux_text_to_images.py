# Imports and setup
from dataclasses import dataclass
from pathlib import Path
import modal
import os
import uuid
from typing import Optional
import time
import sys
import glob

# List of fake users
FAKE_USERS = [
    "default",
    "user1",
    "user2", 
    "user3",
    "user4",
    "user5",
    "user6",
    "user7",
    "user8",
    "user9",
    "user10"
]

# Building up the environment
app = modal.App(name="flux-text-to-images")

# Create base image with common dependencies
base_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate",
    "datasets~=2.13.0",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "gradio~=5.5.0",
    "huggingface-hub",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft==0.11.1",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "starlette==0.41.2",
    "transformers~=4.41.2",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "wandb==0.17.6",
    "boto3",
    "imageio",
    "imageio-ffmpeg",
    "pyyaml",
    "opencv-python",
    "einops",
    "timm",
    "av",
    "spaces"
)

# Create two different images with different diffusers versions
image_dev = base_image.pip_install("diffusers==0.33.1")  # For FLUX.1-dev
# image_stable = base_image.pip_install("diffusers==0.32.0")  # For FLUX.1

# Add ltx_video_distilled directory to both images
image_dev = image_dev.add_local_dir(
    Path(__file__).parent / "ltx_video_distilled",
    remote_path="/root/ltx_video_distilled",
    copy=True
)

image_stable = base_image.add_local_dir(
    Path(__file__).parent / "ltx_video_distilled",
    remote_path="/root/ltx_video_distilled",
    copy=True
)

# Update the GIT_SHA for each version
GIT_SHA_DEV = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # For FLUX.1-dev
GIT_SHA_STABLE = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # For FLUX.1

# # Setup git for both images
# image_dev = (
#     image_dev.apt_install("git", "ffmpeg")
#     .run_commands(
#         "cd /root && git init .",
#         "cd /root && git remote add origin https://github.com/huggingface/diffusers",
#         f"cd /root && git fetch --depth=1 origin {GIT_SHA_DEV} && git checkout {GIT_SHA_DEV}",
#         "cd /root && pip install -e .",
#     )
# )
image_dev = (
    image_dev.apt_install("git", "ffmpeg")
)

image_stable = (
    image_stable.apt_install("git", "ffmpeg")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA_STABLE} && git checkout {GIT_SHA_STABLE}",
        "cd /root && pip install -e .",
    )
)

with image_dev.imports():  # loaded on all of our remote Functions
    import sys
    import os
    from pathlib import Path
    # Add ltx_video_distilled to Python path
    sys.path.append("/root/ltx_video_distilled")
    sys.path.append("/root/ltx_video_distilled/configs")
    # Ensure configs directory exists
    configs_dir = Path("/root/ltx_video_distilled/configs")
    if not configs_dir.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
    
    import torch
    from PIL import Image
    import yaml

with image_stable.imports():
    import sys
    import os
    from pathlib import Path
    
    # Add ltx_video_distilled to Python path
    sys.path.append("/root/ltx_video_distilled")
    sys.path.append("/root/ltx_video_distilled/configs")
    # Ensure configs directory exists
    configs_dir = Path("/root/ltx_video_distilled/configs")
    if not configs_dir.exists():
        configs_dir.mkdir(parents=True, exist_ok=True)
    
    import torch
    from PIL import Image
    import yaml
    
# S3 Configuration
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = '0b762a408ed9101030b7d79189a67410'
S3_SECRET_KEY = '30e1f3a43c87191ff5cb5b829e224b12'

# Initialize S3 client
bucket_url = 'https://eu2.contabostorage.com/'

def get_s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client('s3',
                endpoint_url=bucket_url,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
                config=Config(
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required"
                )
            )

def upload_file(file_name, user_uuid, bucket=S3_BUCKET, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    s3_client = get_s3_client()
    
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    object_name = file_name.split('/')[-1]
    print('object_name', object_name)
    object_name = user_uuid + '/' + object_name
    print('object_name', object_name)
    response = s3_client.upload_file(file_name, bucket, object_name)
    
    # Create a presigned URL for the file
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': object_name},
        ExpiresIn=3600  # URL will be valid for 1 hour
    )   
    print(presigned_url)
    # delete file from local
    os.remove(file_name)
        
    return presigned_url

# Configuration with dataclasses
# Machine learning apps often have a lot of configuration information. We collect up all of our configuration into dataclasses to avoid scattering special/magic values throughout code.
@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "xJnyz"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "mage"
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"
    # User ID for multi-user support
    user_id: str = "default"
    # Whether to use finetuned model or base model
    use_finetuned: bool = True
    use_dev_version: bool = True  # New flag to control which version to use
    
# Storing data created by our app with modal.Volume
volume = modal.Volume.from_name(
    "flux-text-to-images-data", create_if_missing=True
)
MODEL_DIR = "/model"
OUTPUT_PATH = "/output"
USER_MODELS_DIR = "user_models"
USE_WANDB = False

output_volume = modal.Volume.from_name("outputs", create_if_missing=True)


huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

s3_secret = modal.Secret.from_dict({
    "S3_ACCESS_KEY": S3_ACCESS_KEY,
    "S3_SECRET_KEY": S3_SECRET_KEY
})

image_dev = image_dev.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1",
     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # turn on faster downloads from HF
)

image_stable = image_stable.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1",
     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # turn on faster downloads from HF
)   

def get_model_dir(config):
    """Get the appropriate model directory based on user and settings."""
    if not config.use_finetuned:
        return MODEL_DIR
    
    user_model_dir = os.path.join(MODEL_DIR, USER_MODELS_DIR, config.user_id)
    return user_model_dir

@app.function(
    volumes={MODEL_DIR: volume},
    image=image_dev,
    secrets=[huggingface_secret, s3_secret],
    timeout=600,  # 10 minutes
)
def download_models(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    # Always download the base model if it doesn't exist
    if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
        snapshot_download(
            config.model_name,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        )
        DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    
    # Create user directory if it doesn't exist
    if config.use_finetuned:
        user_model_dir = get_model_dir(config)
        os.makedirs(user_model_dir, exist_ok=True)
    
# Load fine-tuning dataset
# Part of the magic of the low-rank fine-tuning is that we only need 3-10 images for fine-tuning. 
# So we can fetch just a few images, stored on consumer platforms like Imgur or Google Drive, 
# whenever we need them — no need for expensive, hard-to-maintain data pipelines.
def load_images(image_urls: list[str]) -> Path:
    import PIL.Image
    from smart_open import open
    # Create a directory to store the images
    img_path = Path("/img")
    # Create the directory if it doesn't exist
    img_path.mkdir(parents=True, exist_ok=True)
    # Loop through the image URLs and download the images
    for ii, url in enumerate(image_urls):
        # Open the image URL and save it to the local directory
        with open(url, "rb") as f:
            # Open the image URL and save it to the local directory
            image = PIL.Image.open(f)
            # Save the image to the local directory
            image.save(img_path / f"{ii}.png")
    # Print the number of images loaded
    print(f"{ii + 1} images loaded")

    return img_path

def clear_memory():
    import torch
    print('clearing memory....')
    # Free up memory
    torch.cuda.empty_cache()
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
    print('memory cleared')
        
# Finetuning with Hugging Face 🧨 Diffusers and Accelerate
@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a cinematic portrait of"
    postfix: str = "in a mysterious foggy forest, glowing amulet around his neck, cinematic lighting"

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117


@app.function(
    image=image_stable,
    gpu="A100-80GB",  # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1800,  # 30 minutes
    secrets=[huggingface_secret, s3_secret]
    + (
        [modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])]
        if USE_WANDB
        else []
    ),
)
def train(instance_example_urls, config):
    # Use the appropriate image based on config
    # if not config.use_dev_version:
    #     # Switch to stable image
    #     train.function.image = image_stable
    
    import subprocess
    from accelerate.utils import write_basic_config
    import shutil
    import time

    # Create user directory if it doesn't exist
    user_model_dir = get_model_dir(config)
    os.makedirs(user_model_dir, exist_ok=True)
    # # Remove pytorch_lora_weights.safetensors
    # rm_old_finetune_model_dir = os.path.join(MODEL_DIR, USER_MODELS_DIR, config.user_id)
    # # Remove old lora weights folders
    # lora_weights_folders = ["lora_weights_John-Body.png"]
    # for folder in lora_weights_folders:
    #     folder_path = os.path.join(rm_old_finetune_model_dir, folder)
    #     if os.path.exists(folder_path):
    #         shutil.rmtree(folder_path)
            
    # If user model doesn't exist, copy base model to user directory
    if not os.path.exists(os.path.join(user_model_dir, "model_index.json")):
        print(f"Copying base model to user directory: {user_model_dir}")
        for item in os.listdir(MODEL_DIR):
            if item == USER_MODELS_DIR:  # Skip the user_models directory
                continue
            s = os.path.join(MODEL_DIR, item)
            d = os.path.join(user_model_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # load data locally
    img_path = load_images(instance_example_urls)
    print('Loaded images')
    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    # Generate timestamp for unique filename
    # timestamp = int(time.time())
    # Get file name from instance_example_urls
    file_name = instance_example_urls[0].split('/')[-1].split('.')[0]
    output_dir = os.path.join(user_model_dir, f"lora_weights_{file_name}")
    os.makedirs(output_dir, exist_ok=True)

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={user_model_dir}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={output_dir}",  # Use the timestamped output directory
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )
    # The trained model information has been output to the volume mounted at user model directory.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()
    clear_memory()

""" 
Running our model
"""
@app.cls(
    image=image_dev,  # Default to dev image
    gpu="A100-80GB", 
    volumes={MODEL_DIR: volume,
             OUTPUT_PATH: output_volume}, 
    secrets=[s3_secret]
)
class Model:
    @modal.enter()
    def load_model(self):
        # Use the appropriate image based on config
        # if not self.config.use_dev_version:
        #     # Switch to stable image
        #     Model.cls.image = image_stable
            
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import login
        import os
        
        # Login to Hugging Face Hub to access gated models
        login(token="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")
        
        # Store models by user_id
        self.user_models = {}
        
        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()
        
        # Load base model
        self.base_pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def get_pipe_for_user(self, config):
        """Get or create a pipeline for the specified user configuration."""
        from diffusers import DiffusionPipeline
        import os
        import glob
        
        # If not using finetuned model, return base model
        if not config.use_finetuned:
            return self.base_pipe
            
        # Check if we already have this user's model loaded
        if config.user_id in self.user_models:
            return self.user_models[config.user_id]
            
        # Get the model directory for this user
        model_dir = get_model_dir(config)
        
        # Check if user model exists
        if not os.path.exists(os.path.join(model_dir, "model_index.json")):
            print(f"User model not found for {config.user_id}, using base model")
            return self.base_pipe
            
        # Load the user's model
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        # Find all LoRA weight directories
        lora_dirs = glob.glob(os.path.join(model_dir, "lora_weights_*"))
        if lora_dirs:
            if config.use_all_models:
                # Load all available LoRA weights
                print(f"Loading all available LoRA weights for user {config.user_id}")
                for lora_dir in lora_dirs:
                    model_name = os.path.basename(lora_dir).replace("lora_weights_", "")
                    print(f"Loading LoRA weights from {lora_dir} with adapter name {model_name}")
                    pipe.load_lora_weights(lora_dir)
            elif config.lora_model_name:
                # If specific model is requested, try to find it
                target_dir = os.path.join(model_dir, f"lora_weights_{config.lora_model_name}")
                if target_dir in lora_dirs:
                    print(f"Loading specified LoRA weights from {target_dir} for user {config.user_id}")
                    pipe.load_lora_weights(target_dir)
                else:
                    print(f"Specified LoRA model {config.lora_model_name} not found, using latest")
                    latest_lora_dir = max(lora_dirs, key=os.path.getmtime)
                    pipe.load_lora_weights(latest_lora_dir)
            else:
                # If no specific model requested, use latest
                latest_lora_dir = max(lora_dirs, key=os.path.getmtime)
                print(f"Loading latest LoRA weights from {latest_lora_dir} for user {config.user_id}")
                pipe.load_lora_weights(latest_lora_dir)
        else:
            print(f"No LoRA weights found for user {config.user_id}")
            
        # Cache the model
        self.user_models[config.user_id] = pipe
        return pipe

    @modal.method()
    def inference(self, text, config):
        import tempfile
        import PIL.Image
        import io
        clear_memory()
        # Get the appropriate pipeline for this user
        pipe = self.get_pipe_for_user(config)
        
        image = pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]
        
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Generate a unique ID for the image
        user_uuid = config.user_id if config.user_id != "default" else str(uuid.uuid4())
        
        # Upload to S3 and get the URL
        image_url = upload_file(temp_file_path, user_uuid)
        clear_memory()
        return image_url
 
    @modal.method()
    def image_to_video(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        width: Optional[int] = 768,
        height: Optional[int] = 512,
        improve_texture_flag: bool = True,
    ):
        """Generate a video from an image using text prompt."""
        import sys
        from ltx_video_distilled.app import generate
        import tempfile
        import os
        print('input seed ', seed)
        
        clear_memory()
        random_seed = seed if seed else int(time.time())
        seed = random_seed if seed is None else seed
        print('random seed ', random_seed)
        
        # Create a temporary file to store the input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            input_image_path = temp_file.name
            
        try:
            # Generate video using the generate function
            video_path, _seed = generate(
                prompt=prompt,
                negative_prompt=negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted",
                input_image_filepath=input_image_path,
                input_video_filepath=None,
                height_ui=height,
                width_ui=width,
                mode="image-to-video",
                duration_ui=num_frames/30 if num_frames else 2,  # Convert frames to duration in seconds
                ui_frames_to_use=num_frames,
                seed_ui=seed,
                randomize_seed=seed is None,
                ui_guidance_scale=1,
                improve_texture_flag=improve_texture_flag
            )
            
            # Upload to S3 and get the URL
            user_uuid = "default-test-img-2-video"
            video_url = upload_file(video_path, user_uuid)
            clear_memory()
            return video_url
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_image_path):
                os.remove(input_image_path)

def slugify(s: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{''.join(c if c.isalnum() else '-' for c in s[:100]).strip('-')}.mp4"




@app.local_entrypoint()
def run(  # add more config params here to make training configurable
    max_train_steps: int = 250,
    user_id: str = "default",
    instance_name: str = "xJnyz",
    class_name: str = "mage",
):
    print("🎨 loading model")
    config = SharedConfig(
        user_id=user_id,
        instance_name=instance_name,
        class_name=class_name
    )
    download_models.remote(config)
    print("🎨 setting up training")
    train_config = TrainConfig(
        max_train_steps=max_train_steps,
        user_id=user_id,
        instance_name=instance_name,
        class_name=class_name
    )
    instance_example_urls = (
        Path(TrainConfig.instance_example_urls_file).read_text().splitlines()
    )
    train.remote(instance_example_urls, train_config)
    print("🎨 training finished")



"""
FOR UI

"""

@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 6
    lora_model_name: Optional[str] = None  # Add field to specify which LoRA model to use
    use_all_models: bool = False  # Add field to specify if all models should be used


web_image = image_dev.add_local_dir(
    # Add local web assets to the image
    Path(__file__).parent / "assets",
    remote_path="/assets",
    copy=True
)

@app.function(
    image=web_image,
    max_containers=1,
    secrets=[s3_secret],
    volumes={MODEL_DIR: volume},
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from fastapi import FastAPI, Request, Form, Cookie
    from fastapi.responses import FileResponse, RedirectResponse
    from gradio.routes import mount_gradio_app
    import json
    import glob
    import os
    
    # download_models.remote(SharedConfig())
    web_app = FastAPI()

    # Function to get available LoRA models for a user
    @web_app.get("/api/lora_models")
    def get_available_lora_models(user_id):
        try:
            model_dir = os.path.join(MODEL_DIR, USER_MODELS_DIR, user_id)
            lora_dirs = glob.glob(os.path.join(model_dir, "lora_weights_*"))
            if lora_dirs:
                # Extract model names from directory names
                model_names = [os.path.basename(d).replace("lora_weights_", "") for d in lora_dirs]
                return model_names
            return []
        except:
            return []

    # User management routes
    @web_app.get("/api/users", response_model=list)
    def get_users():
        # Return fake users list
        return FAKE_USERS

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text="", user_id="default", lora_model_name=None):
        if not text:
            text = example_prompts[0]
        
        # Create config with user settings
        user_config = AppConfig(
            user_id=user_id,
            use_finetuned=True,  # Always use finetuned model
            lora_model_name=lora_model_name
        )
        
        # Generate 4 images instead of 1
        image_urls = []
        for _ in range(4):
            image_urls.append(Model().inference.remote(text, user_config))
        return image_urls
    
    # Function to start training from UI
    def start_training(user_id, instance_name, class_name, image_urls, max_train_steps, prefix, postfix):
        if not user_id or user_id == "new_user":
            return "Please select or create a valid user ID"
        
        if not instance_name:
            instance_name = "xJnyz"
            
        if not class_name:
            class_name = "mage"
            
        if not image_urls:
            return "Please provide image URLs for training"
            
        # Parse image URLs
        urls = [url.strip() for url in image_urls.split("\n") if url.strip()]
        if not urls:
            return "Please provide valid image URLs for training"
            
        try:
            train_steps = int(max_train_steps)
        except:
            train_steps = 250
            
        # Create config
        config = SharedConfig(
            user_id=user_id,
            instance_name=instance_name,
            class_name=class_name
        )
        # if model is not downloaded, download it
        if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
            print("Downloading model.......")
            download_models.remote(config)
            print("Model downloaded")
            
        # Setup training config
        train_config = TrainConfig(
            max_train_steps=train_steps,
            user_id=user_id,
            instance_name=instance_name,
            class_name=class_name,
            prefix=prefix,
            postfix=postfix
        )
        
        # Start training
        train.remote(urls, train_config)
        return f"Training started for user {user_id} with {len(urls)} images. This may take some time."

    # set up AppConfig
    config = AppConfig()

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    modal_docs_url = "https://modal.com/docs"
    modal_example_url = f"{modal_docs_url}/examples/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

### Learn how to make a "Dreambooth" for your own pet [here]({modal_example_url}).
    """

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    # add a gradio UI around inference
    with gr.Blocks(
        theme=theme,
        css=css,
        title=f"Generate images of {config.instance_name} on Modal",
    ) as interface:
        # Store user settings
        user_id = gr.State("default")
        selected_model = gr.State(None)
        
        gr.Markdown(
            f"# Generate images of {instance_phrase}.\n\n{description}",
        )
        
        with gr.Tabs():
            with gr.TabItem("Generate Images"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # User settings
                        user_dropdown = gr.Dropdown(
                            label="Select User",
                            choices=get_users(),
                            value=get_users()[0]
                        )
                        lora_model_dropdown = gr.Dropdown(
                            label="Select LoRA Model",
                            choices=[],
                            value=None,
                            visible=True
                        )
                        
                        # Update user dropdown when page loads
                        def update_user_list():
                            return gr.Dropdown(choices=FAKE_USERS)
                        
                        # Update LoRA model dropdown when user changes
                        def update_lora_models(user_id):
                            print(f"Updating models for user: {user_id}")
                            models = get_available_lora_models(user_id)
                            print(f"Available models: {models}")
                            
                            if models:
                                return gr.Dropdown(choices=models, value=models[0], visible=True)
                            return gr.Dropdown(choices=[], value=None, visible=True)
                        
                        # Update selected model when LoRA dropdown changes
                        def update_selected_model(model_name):
                            return model_name
                            
                        user_dropdown.change(
                            lambda x: x,
                            inputs=[user_dropdown], 
                            outputs=[user_id]
                        ).then(
                            update_lora_models,
                            inputs=[user_id],
                            outputs=[lora_model_dropdown]
                        )
                        
                        lora_model_dropdown.change(
                            update_selected_model,
                            inputs=[lora_model_dropdown],
                            outputs=[selected_model]
                        )

                    with gr.Column(scale=3):
                        inp = gr.Textbox(  # input text component
                            label="",
                            placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                            lines=10,
                        )
                        # Change to Gallery to display multiple images using S3 URLs
                        out = gr.Gallery(  
                            label="", elem_id="output", columns=2, rows=2, height=512,
                            object_fit="contain"
                        )
                
                with gr.Row():
                    btn = gr.Button("Dream", variant="primary", scale=2)
                    btn.click(
                        fn=go, 
                        inputs=[inp, user_id, selected_model], 
                        outputs=out
                    )  # connect inputs and outputs with inference function

                with gr.Column(variant="compact"):
                    # add in a few examples to inspire users
                    for ii, prompt in enumerate(example_prompts):
                        btn = gr.Button(prompt, variant="secondary")
                        btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)
            
            with gr.TabItem("Image to Video"):
                with gr.Row():
                    with gr.Column():
                        i2v_image_input = gr.Image(
                            label="Upload or select an image to animate",
                            type="filepath"
                        )
                        i2v_prompt = gr.Textbox(
                            label="Video prompt",
                            placeholder="Describe the motion or animation you want to see",
                            lines=3
                        )
                        i2v_negative_prompt = gr.Textbox(
                            label="Negative prompt (optional)",
                            placeholder="What you don't want to see in the video",
                            lines=2
                        )
                        with gr.Row():
                            i2v_duration = gr.Number(
                                label="Duration (seconds)",
                                value=2,
                                minimum=1,
                                maximum=8,
                                step=1
                            )
                            i2v_num_steps = gr.Number(
                                label="Number of inference steps",
                                value=50,
                                minimum=1,
                                maximum=100
                            )
                        with gr.Row():
                            i2v_width = gr.Number(
                                label="Width",
                                value=768,
                                minimum=256,
                                maximum=1280,
                                step=32
                            )
                            i2v_height = gr.Number(
                                label="Height",
                                value=512,
                                minimum=256,
                                maximum=1280,
                                step=32
                            )
                        with gr.Row():
                            i2v_seed = gr.Number(
                                label="Seed (optional)",
                                value=None
                            )
                            i2v_random_seed = gr.Checkbox(
                                label="Random seed",
                                value=True
                            )
                        i2v_improve_texture = gr.Checkbox(
                            label="Improve texture quality",
                            value=True,
                            info="Enable to improve texture quality in the generated video"
                        )
                        i2v_video_output = gr.Video(
                            label="Generated Video"
                        )
                
                with gr.Row():
                    i2v_btn = gr.Button("Generate Video", variant="primary", scale=2)
                    i2v_btn.click(
                        fn=lambda img, prompt, neg_prompt, duration, steps, seed, random_seed, width, height, improve_texture: 
                            Model().image_to_video.remote(
                                image_bytes=open(img, 'rb').read() if img else None,
                                prompt=prompt,
                                negative_prompt=neg_prompt,
                                num_frames=int(duration * 30) if duration else None,  # Convert duration to frames
                                num_inference_steps=steps,
                                seed=None if random_seed else seed,
                                width=width,
                                height=height,
                                improve_texture_flag=improve_texture
                            ) if img else None,
                        inputs=[
                            i2v_image_input,
                            i2v_prompt,
                            i2v_negative_prompt,
                            i2v_duration,
                            i2v_num_steps,
                            i2v_seed,
                            i2v_random_seed,
                            i2v_width,
                            i2v_height,
                            i2v_improve_texture
                        ],
                        outputs=i2v_video_output
                    )
            
            with gr.TabItem("Train Your Model"):
                with gr.Row():
                    with gr.Column():
                        train_user_dropdown = gr.Dropdown(
                            label="Select User",
                            choices=FAKE_USERS,
                            value=FAKE_USERS[0]
                        )
                        train_instance_name = gr.Textbox(
                            label="Instance Name",
                            placeholder="Name of your character/object - Nên chọn các ký tự đặc biệt, không có trong tiếng anh, ví dụ qwert, aasjjwk - Để định danh cho nhân vật",
                            value="Should be xJnyz"
                        )
                        train_class_name = gr.Textbox(
                            label="Class Name",
                            placeholder="Type of character/object (e.g., mage, cat, robot, man, woman, etc.)",
                            value="mage"
                        )
                        train_prefix = gr.Textbox(
                            label="Prompt Prefix",
                            placeholder="Text that comes before the instance name in training prompt",
                            value="a cinematic portrait of"
                        )
                        train_postfix = gr.Textbox(
                            label="Prompt Postfix",
                            placeholder="Text that comes after the instance name in training prompt",
                            value="in a mysterious foggy forest, glowing amulet around his neck, cinematic lighting"
                        )
                        train_model_name = gr.Textbox(
                            label="Model Name",
                            placeholder="Enter a name for your model (e.g., my_character_v1)",
                            value=""
                        )
                        train_image_urls = gr.Textbox(
                            label="Image URLs",
                            placeholder="Enter image URLs (one per line) for training",
                            lines=5
                        )
                        train_steps = gr.Number(
                            label="Training Steps",
                            value=250,
                            minimum=100,
                            maximum=1000,
                            step=50
                        )
                        train_btn = gr.Button("Start Training", variant="primary")
                        train_output = gr.Textbox(label="Training Status")
                        
                        # Update user dropdown for training tab
                        def update_train_user_list():
                            return gr.Dropdown(choices=FAKE_USERS)
                        
                        train_user_dropdown.change(
                            lambda x: x,
                            inputs=[train_user_dropdown],
                            outputs=[user_id]
                        )
                        
                        # Connect training button
                        train_btn.click(
                            fn=start_training,
                            inputs=[
                                train_user_dropdown,
                                train_instance_name,
                                train_class_name,
                                train_image_urls,
                                train_steps,
                                train_prefix,
                                train_postfix,
                                train_model_name
                            ],
                            outputs=train_output
                        )

        # Update user lists when interface loads
        interface.load(update_user_list, outputs=[user_dropdown])
        interface.load(update_train_user_list, outputs=[train_user_dropdown])

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )
