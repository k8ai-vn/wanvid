# Imports and setup
from dataclasses import dataclass
from pathlib import Path
import modal
import os
import uuid
from typing import Optional
import time
import sys

# Building up the environment
app = modal.App(name="flux-text-to-images")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
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
    "diffusers==0.33.1",
    "pyyaml",
    "opencv-python",
    "einops",
    "timm",
    "av",
    "spaces"
)

# Add ltx_video_distilled directory to the image
image = image.add_local_dir(
    Path(__file__).parent / "ltx_video_distilled",
    remote_path="/root/ltx_video_distilled",
    copy=True
)

with image.imports():  # loaded on all of our remote Functions
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

# ### Downloading scripts and installing a git repo with `run_commands`

# We'll use an example script from the `diffusers` library to train the model.
# We acquire it from GitHub and install it in our environment with a series of commands.
# The container environments Modal Functions run in are highly flexible --
# see [the docs](https://modal.com/docs/guide/custom-container) for more details.

GIT_SHA = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch

image = (
    image.apt_install("git", "ffmpeg")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        # "cd /root && pip install -e .",
    )
)

# S3 Configuration
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = ''
S3_SECRET_KEY = ''

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

image = image.env(
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
    image=image,
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
    image=image,
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
    import subprocess
    from accelerate.utils import write_basic_config
    import shutil

    # Create user directory if it doesn't exist
    user_model_dir = get_model_dir(config)
    os.makedirs(user_model_dir, exist_ok=True)
    
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

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

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
            f"--output_dir={user_model_dir}",
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
@app.cls(image=image, 
         gpu="A100-80GB", 
         volumes={MODEL_DIR: volume,
                  OUTPUT_PATH: output_volume}, 
         secrets=[s3_secret])
class Model:
    @modal.enter()
    def load_model(self):
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
        
        # Check for LoRA weights
        weight_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors") or f.endswith(".bin")]
        print(f'weight_files for user {config.user_id}:', weight_files)
        if len(weight_files) > 2:
            print(f"Loading LoRA weights from {weight_files[0]} for user {config.user_id}")
            pipe.load_lora_weights(model_dir)
        else:
            print(f"Skipping LoRA weight loading for user {config.user_id}. Found {len(weight_files)} weight files.")
            
        # Cache the model
        self.user_models[config.user_id] = pipe
        return pipe

    @modal.method()
    def inference(self, text, config):
        import tempfile
        import PIL.Image
        import io
        
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


web_image = image.add_local_dir(
    # Add local web assets to the image
    Path(__file__).parent / "assets",
    remote_path="/assets",
    copy=True
)

@app.function(
    image=web_image,
    max_containers=1,
    secrets=[s3_secret],
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from fastapi import FastAPI, Request, Form, Cookie
    from fastapi.responses import FileResponse, RedirectResponse
    from gradio.routes import mount_gradio_app
    import json
    
    # download_models.remote(SharedConfig())
    web_app = FastAPI()

    # Call out to the inference in a separate Modal environment with a GPU
    def go(text="", user_id="default", use_finetuned=True):
        if not text:
            text = example_prompts[0]
        
        # Create config with user settings
        user_config = AppConfig(
            user_id=user_id,
            use_finetuned=use_finetuned
        )
        
        # Generate 4 images instead of 1
        image_urls = []
        for _ in range(4):
            image_urls.append(Model().inference.remote(text, user_config))
        return image_urls
    
    # Function to upload image to S3 and then generate video
    def generate_video(image, prompt, negative_prompt=None, num_frames=None, num_inference_steps=None, seed=None, width=768, height=512):
        if not image:
            return None
        
        if not prompt:
            prompt = "A short video clip"
            
        # Read image file as bytes
        with open(image, 'rb') as f:
            image_bytes = f.read()
            
        # Call the image_to_video method
        video_url = Model().image_to_video.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height
        )
        return video_url

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

    # User management routes
    @web_app.get("/api/users", response_model=list)
    async def get_users():
        # List directories in USER_MODELS_DIR to get available users
        try:
            user_models_path = os.path.join(MODEL_DIR, USER_MODELS_DIR)
            user_dirs = [d for d in os.listdir(user_models_path) 
                        if os.path.isdir(os.path.join(user_models_path, d))]
            return user_dirs
        except FileNotFoundError:
            return []

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
        use_finetuned = gr.State(True)
        
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
                            choices=["default", "new_user"],
                            value="default"
                        )
                        new_user_input = gr.Textbox(
                            label="New User ID",
                            placeholder="Enter a unique ID for new user",
                            visible=False
                        )
                        use_finetuned_checkbox = gr.Checkbox(
                            label="Use Finetuned Model",
                            value=True
                        )
                        
                        # Update user dropdown when page loads
                        def update_user_list():
                            try:
                                user_models_path = os.path.join(MODEL_DIR, USER_MODELS_DIR)
                                users = ["default", "new_user"] + [d for d in os.listdir(user_models_path) 
                                        if os.path.isdir(os.path.join(user_models_path, d)) and d != "default"]
                                return gr.Dropdown(choices=list(set(users)))
                            except FileNotFoundError:
                                return gr.Dropdown(choices=["default", "new_user"])
                        
                        # Show/hide new user input based on dropdown selection
                        def toggle_new_user_input(choice):
                            if choice == "new_user":
                                return gr.Textbox(visible=True), "default"
                            else:
                                return gr.Textbox(visible=False), choice
                        
                        user_dropdown.change(
                            toggle_new_user_input,
                            inputs=[user_dropdown],
                            outputs=[new_user_input, user_id]
                        )
                        
                        # Update user_id when new user is created
                        def update_user_id(new_id):
                            if new_id and new_id.strip():
                                return new_id.strip()
                            return "default"
                        
                        new_user_input.change(
                            update_user_id,
                            inputs=[new_user_input],
                            outputs=[user_id]
                        )
                        
                        # Update use_finetuned state
                        use_finetuned_checkbox.change(
                            lambda x: x,
                            inputs=[use_finetuned_checkbox],
                            outputs=[use_finetuned]
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
                        inputs=[inp, user_id, use_finetuned], 
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
                            choices=["default", "new_user"],
                            value="default"
                        )
                        train_new_user_input = gr.Textbox(
                            label="New User ID",
                            placeholder="Enter a unique ID for new user",
                            visible=False
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
                            try:
                                user_models_path = os.path.join(MODEL_DIR, USER_MODELS_DIR)
                                users = ["default", "new_user"] + [d for d in os.listdir(user_models_path) 
                                        if os.path.isdir(os.path.join(user_models_path, d)) and d != "default"]
                                return gr.Dropdown(choices=list(set(users)))
                            except FileNotFoundError:
                                return gr.Dropdown(choices=["default", "new_user"])
                        
                        # Show/hide new user input based on dropdown selection for training
                        def toggle_train_new_user_input(choice):
                            if choice == "new_user":
                                return gr.Textbox(visible=True)
                            else:
                                return gr.Textbox(visible=False)
                        
                        train_user_dropdown.change(
                            toggle_train_new_user_input,
                            inputs=[train_user_dropdown],
                            outputs=[train_new_user_input]
                        )
                        
                        # Define a function to get the user ID for training
                        def get_train_user_id(dropdown_choice, new_user_id):
                            if dropdown_choice == "new_user" and new_user_id and new_user_id.strip():
                                return new_user_id.strip()
                            elif dropdown_choice != "new_user":
                                return dropdown_choice
                            return "default"
                        
                        # Connect training button - using direct values instead of lambda function
                        train_btn.click(
                            fn=start_training,
                            inputs=[
                                train_user_dropdown,
                                train_instance_name,
                                train_class_name,
                                train_image_urls,
                                train_steps,
                                train_prefix,
                                train_postfix
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
