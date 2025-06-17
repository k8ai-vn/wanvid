import time
from io import BytesIO
from pathlib import Path
import os
import modal
import fastapi
import base64
import random
from pydantic import BaseModel
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"
NUM_INFERENCE_STEPS = 12
cuda_dev_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])

cuda_dev_image = cuda_dev_image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1",
     "XFORMERS_ENABLE_TRITON": "1",
     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # turn on faster downloads from HF
)

cuda_tts_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
).entrypoint([])

class ModelConfig:
    model_name: str = "black-forest-labs/FLUX.1-dev"
    # model_name: str = "black-forest-labs/FLUX.1-schnell"
    adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
    transformer_framepack = "lllyasviel/FramePack_F1_I2V_HY_20250503"
    feature_extractor_framepack = "lllyasviel/flux_redux_bfl"
    image_encoder_framepack = "lllyasviel/flux_redux_bfl"
    model_hunyuan = "hunyuanvideo-community/HunyuanVideo"
    model_erax_ai = "erax-ai/EraX-Smile-UnixSex-F5"
    
flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.52.3",
        "huggingface_hub[hf_transfer]",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        # f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "git+https://github.com/huggingface/diffusers",
        "numpy<2",
        "fastapi[standard]",
        "optimum-quanto",
        "numpy",
        "peft",
        "imageio",
        "einops",
        "torchvision",
        "av==12.1.0",
        "matplotlib",
        "vinorm",
        "f5-tts",
        "librosa"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)
flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

tts_image = (
    cuda_tts_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "gradio==4.25",
        "fastapi[standard]",
        "deepfilternet==0.5.6",
        "vinorm==2.0.7",
        "underthesea==6.8.0",
        "deepspeed",
        "cutlet",
        "unidic",
        "torch",
        "torchaudio",
        "huggingface_hub[hf_transfer]",
        "underthesea",
        "unidecode",
        "coqpit",
        "numpy==1.22.0;python_version<='3.10'",
        "numpy>=1.24.3;python_version>'3.10'",
        "cython>=0.29.30", 
        "scipy>=1.11.2",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "scikit-learn>=1.3.0",
        "numba==0.55.1;python_version<'3.9'",
        "numba>=0.57.0;python_version>='3.9'",
        "inflect>=5.6.0",
        "tqdm>=4.64.1",
        "anyascii>=0.3.0",
        "pyyaml>=6.0",
        "fsspec>=2023.6.0",
        "aiohttp>=3.8.1",
        "packaging>=23.1",
        "mutagen==1.47.0",
        "flask>=2.0.1",
        "pysbd>=0.3.4",
        "umap-learn>=0.5.1",
        "pandas>=1.4,<2.0",
        "matplotlib>=3.7.0",
        "trainer>=0.0.36",
        "jieba",
        "pypinyin",
        "hangul_romanize",
        "gruut[de,es,fr]==2.2.3",
        "jamo",
        "nltk", 
        "g2pkk>=0.1.1",
        "bangla",
        "bnnumerizer",
        "bnunicodenormalizer",
        "einops>=0.6.0",
        "transformers==4.45.2",
        "encodec>=0.1.1",
        "num2words",
        "spacy[ja]>=3"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

# Storing data created by our app with modal.Volume
volume = modal.Volume.from_name(
    "flux-text-to-images-data", 
    create_if_missing=True
)
volume_adapter = modal.Volume.from_name(
    "flux-text-to-images-data-adapter", 
    create_if_missing=True
)

volume_transformer_framepack =  modal.Volume.from_name(
    "framepack-F1_I2V", 
    create_if_missing=True
)
volume_feature_extractor_framepack = modal.Volume.from_name(
    "framepack-feature-extractor",
    create_if_missing=True
)
volume_image_encoder_framepack = modal.Volume.from_name(
    "framepack-image-encoder",
    create_if_missing=True
)
volume_hunyuan_commutity = modal.Volume.from_name(
    "model-hunyuan-commutity",
    create_if_missing=True

)
volume_erax_ai = modal.Volume.from_name(
    "model-erax-ai",
    create_if_missing=True
)

volume_vixtts = modal.Volume.from_name(
    "model-vixtts",
    create_if_missing=True
)
# Add diffusers_helper directory to both images
flux_image = flux_image.add_local_dir(
    Path(__file__).parent / "FramePack_F1_ok",
    remote_path="/root/FramePack_F1_ok",
    copy=True
)

flux_image = flux_image.add_local_file(
    Path(__file__).parent / "f5tts_wrapper.py",
    remote_path="/root/f5tts_wrapper.py",
    copy=True
)

flux_image = flux_image.add_local_dir(
    Path(__file__).parent / "vixtts_demo",
    remote_path="/root/vixtts_demo",
    copy=True
)

tts_image = tts_image.add_local_dir(
    Path(__file__).parent / "vixtts_demo",
    remote_path="/root/vixtts_demo",
    copy=True
)

MODEL_DIR = "/models"
MODEL_ADAPTER_DIR = "/adapter"
MODEL_TRANSFORMER_FRAMEPACK_DIR = "/transformer_framepack"
MODEL_FEATURE_EXTRACTOR_FRAMEPACK_DIR = "/feature_extractor_framepack"
MODEL_IMAGE_ENCODER_FRAMEPACK_DIR = "/image_encoder_framepack"
MODEL_HUNYUAN_COMMUNITY_DIR = "/hunyuan_community"
MODEL_ERAX_AI_DIR = "/erax_ai"
MODEL_VIXTTS_DIR = "/vixtts"

app = modal.App("ai-gen-flux", image=flux_image)
# app_tts = modal.App("ai-gen-tts", image=tts_image)


with flux_image.imports():
    import diffusers
    import torch
    import sys
    sys.path.append("/root/FramePack_F1_ok")
    sys.path.append("/root/f5tts_wrapper.py")
    sys.path.append("/root/vixtts_demo")
    
with tts_image.imports():
    import sys
    import torch
    sys.path.append("/root/vixtts_demo")
    sys.path.append("/root/vixtts_demo/TTS")
    
class T2IRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_images: int = 6
class I2VRequest(BaseModel):
    prompt: str
    image_url: str
    width: int = 1024
    height: int = 1024
    num_images: int = 91
    num_inference_steps: int = 25
    guidance_scale: float = 10.0
    latent_window_size: int = 9
    total_second_length: float = 2.0
    use_teacache: bool = True
    gpu_memory_preservation: float = 6.0
    mp4_crf: int = 16
    
@app.function(volumes={MODEL_DIR: volume, 
                       MODEL_ADAPTER_DIR: volume_adapter,
                       MODEL_TRANSFORMER_FRAMEPACK_DIR: volume_transformer_framepack,
                       MODEL_FEATURE_EXTRACTOR_FRAMEPACK_DIR: volume_feature_extractor_framepack,
                       MODEL_IMAGE_ENCODER_FRAMEPACK_DIR: volume_image_encoder_framepack,
                       MODEL_HUNYUAN_COMMUNITY_DIR: volume_hunyuan_commutity,
                       MODEL_ERAX_AI_DIR: volume_erax_ai,
                       MODEL_VIXTTS_DIR: volume_vixtts}, 
              secrets=[modal.Secret.from_name("huggingface")])
def download_model():
    from huggingface_hub import snapshot_download
    from huggingface_hub import login
    login(token=os.environ["hg_token"])
    # snapshot_download(
    #     ModelConfig.model_name,
    #     local_dir=MODEL_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
    # snapshot_download(
    #     ModelConfig.adapter_id,
    #     local_dir=MODEL_ADAPTER_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
    # snapshot_download(
    #     ModelConfig.transformer_framepack,
    #     local_dir=MODEL_TRANSFORMER_FRAMEPACK_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
    
    # snapshot_download(
    #     ModelConfig.feature_extractor_framepack,
    #     local_dir=MODEL_FEATURE_EXTRACTOR_FRAMEPACK_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
    
    # snapshot_download(
    #     ModelConfig.image_encoder_framepack,
    #     local_dir=MODEL_IMAGE_ENCODER_FRAMEPACK_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
        
    # snapshot_download(
    #     ModelConfig.model_hunyuan,
    #     local_dir=MODEL_HUNYUAN_COMMUNITY_DIR,
    #     ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    # )
    snapshot_download(
        ModelConfig.model_erax_ai,
        local_dir=MODEL_ERAX_AI_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    
def optimize(pipe, compile=True):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compilation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=NUM_INFERENCE_STEPS,  # use ~50 for [dev], smaller for [schnell]
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe


@app.cls(gpu="L40S", 
         timeout=3600, secrets=[modal.Secret.from_name("huggingface")], 
         max_containers=10,
         volumes={MODEL_DIR: volume,
                  MODEL_ADAPTER_DIR: volume_adapter,
            "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
            "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
            "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
            "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True,
        ),
    }, enable_memory_snapshot=True)
class Model:
    compile: bool = (  # see section on torch.compile below for details
        modal.parameter(default=False)
    )

    @modal.enter(snap=True)
    def enter(self):
        self.pipe = diffusers.FluxPipeline.from_pretrained(
            MODEL_DIR, 
            torch_dtype=torch.bfloat16  
        )
        # self.pipe.enable_model_cpu_offload()
        
    @modal.enter(snap=False)
    def setup(self):
        self.pipe.to("cuda")  # Move the model to a GPU!
        self.pipe.load_lora_weights(MODEL_ADAPTER_DIR)
        self.pipe.fuse_lora()
        self.pipe = optimize(self.pipe, compile=self.compile)

        
    @modal.method()
    def t2i_inference(self, prompt: str, width: int = 1024, 
                      height: int = 1024, num_images: int = 6) -> bytes:
        import numpy as np
        print("Generating images...")
        device = "cuda"
        # random seed
        MAX_SEED = np.iinfo(np.int32).max
        
        images = []
        for i in range(num_images):
            seed = random.randint(0, MAX_SEED)
            generator = torch.Generator(device=device).manual_seed(int(float(seed)))
            image = self.pipe(
                prompt,
                width=width,
                height=height,
                output_type="pil", 
                generator=generator,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=3.5,
                max_sequence_length=512,
            ).images[0]
            
            byte_stream = BytesIO()
            image.save(byte_stream, format="JPEG")
            images.append(base64.b64encode(byte_stream.getvalue()).decode("utf-8"))

        # Return all images
        return {
            f"image{i+1}": images[i] for i in range(num_images)
        }

@app.function(
         gpu="L40S", 
         image=flux_image,
         timeout=3600, 
         secrets=[modal.Secret.from_name("huggingface")], 
         max_containers=10,
         volumes={MODEL_TRANSFORMER_FRAMEPACK_DIR: volume_transformer_framepack,
                  MODEL_FEATURE_EXTRACTOR_FRAMEPACK_DIR: volume_feature_extractor_framepack,
                  MODEL_IMAGE_ENCODER_FRAMEPACK_DIR: volume_image_encoder_framepack,
                  MODEL_HUNYUAN_COMMUNITY_DIR: volume_hunyuan_commutity,
                  "/root/outputs": modal.Volume.from_name("outputs", create_if_missing=True),
                  "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
                  "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
                  "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
                  "/root/.inductor-cache": modal.Volume.from_name( "inductor-cache", create_if_missing=True),
         }
)
def i2v_inference( 
                    prompt: str,
                    image_url: str, 
                    width: int = 1024, 
                    height: int = 1024, 
                    num_inference_steps: int = 25,
                    guidance_scale: float = 10.0,
                    latent_window_size: int = 9,
                    total_second_length: float = 2.0,
                    use_teacache: bool = True,
                    gpu_memory_preservation: float = 6.0,
                    mp4_crf: int = 16) -> bytes:
    import io
    import base64
    import numpy as np
    from diffusers.utils import load_image
    from diffusers_helper.utils import resize_and_center_crop
    from diffusers_helper.bucket_tools import find_nearest_bucket
    from diffusers_helper.thread_utils import AsyncStream
    from FramePack_F1_ok.app import worker as worker_framepack

    # Load and process input image
    input_image = load_image(image_url)
    input_image_np = np.array(input_image)
    
    # Find nearest bucket size for the image
    H, W, C = input_image_np.shape
    height, width = find_nearest_bucket(H, W, resolution=640)
    input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

    # Set up parameters for worker
    n_prompt = ""  # Empty negative prompt
    seed = 31337  # Default seed
    cfg = 1.0  # Default cfg
    rs = 0.0  # Default rs
    steps = num_inference_steps
    gs = guidance_scale

    # # Create AsyncStream instance
    # stream = AsyncStream()
    # Call worker function
    print('promp', prompt)
    print('seed', seed)
    print('total_second_length', total_second_length)
    print('steps', steps)
    result = worker_framepack(input_image=input_image_np, 
            prompt=prompt, 
            n_prompt=n_prompt, 
            seed=seed, 
            total_second_length=total_second_length, 
            latent_window_size=latent_window_size, 
            steps=steps, 
            cfg=cfg, 
            gs=gs, 
            rs=rs, 
            gpu_memory_preservation=gpu_memory_preservation, 
            use_teacache=use_teacache, 
            mp4_crf=mp4_crf)
    return result
        # # Get output file path from worker's output queue
        # output_filename = None
        # while True:
        #     flag, data = stream.output_queue.next()
        #     if flag == 'file':
        #         output_filename = data
        #         break
        #     elif flag == 'end':
        #         break

        # # Read the output file and convert to base64
        # if output_filename:
        #     with open(output_filename, 'rb') as f:
        #         video_base64 = base64.b64encode(f.read())
        #     return video_base64
        # else:
        #     raise Exception("Failed to generate video")

@app.function(gpu="T4", 
                    image=tts_image,
                    timeout=3600, 
                    secrets=[modal.Secret.from_name("huggingface")], 
                    max_containers=10,
                    volumes={MODEL_VIXTTS_DIR: volume_vixtts,
                            "/root/outputs": modal.Volume.from_name("outputs", create_if_missing=True)})
def create_audio_f5tts(text_to_generate: str, audio_url: str):
    # from vinorm import TTSnorm # Gotta normalize that Vietnamese text first
    # from f5tts_wrapper import F5TTSWrapper # Our handy wrapper class
    
    # eraX_ckpt_path = f"{MODEL_ERAX_AI_DIR}/models/model_42000.safetensors" # <-- CHANGE THIS!
    # vocab_file = f"{MODEL_ERAX_AI_DIR}/models/vocab.txt" # <-- CHANGE THIS!

    # Save audio url to temp file
    # audio_path = "audio.wav"
    # # ref_audio_path = audio_path
    # with open(audio_path, "wb") as f:
    #     f.write(requests.get(audio_url).content)
        
    from vixtts_demo.vixtts_demo import run_tts, load_model
    
    metrics_text, audio_base64 = run_tts(lang='vi', 
                                        tts_text=text_to_generate, 
                                        speaker_audio_file=audio_url, 
                                        normalize_text=True,
                                        model_dir=MODEL_VIXTTS_DIR)
    # print('metrics_text', metrics_text)
    # print('text_to_generate', text_to_generate)
    # with open(out_path, "rb") as f:
    #     audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    return dict(audio=audio_base64)
    
    # # --- Let's Go! ---
    # print("Initializing the TTS engine... (Might take a sec)")
    # tts = F5TTSWrapper(
    #     model_name="F5TTS_v1_Base",
    #     vocoder_name="vocos",
    #     ckpt_path=eraX_ckpt_path,
    #     vocab_file=vocab_file,
    #     use_ema=True,
    #     target_sample_rate=24000,
    #     n_mel_channels = 100,
    #     hop_length = 256,
    #     win_length = 1024,
    #     n_fft      = 1024,
    #     ode_method = 'euler',
    # )
    # # Normalize the reference text (makes it easier for the model)
    # ref_text_norm = TTSnorm(ref_text)
    # print("Processing the reference voice...")
    # # Feed the model the reference voice ONCE
    # # Provide ref_text for better quality, or set ref_text="" to use Whisper for auto-transcription (if installed)
    # tts.preprocess_reference(
    #     ref_audio_path=ref_audio_path,
    #     ref_text=ref_text_norm,
    #     clip_short=True # Keeps reference audio to a manageable length (~12s)
    # )
    # print(f"Reference audio duration used: {tts.get_current_audio_length():.2f} seconds")
    # # --- Generate New Speech ---
    # print("Generating new speech with the cloned voice...")
    # # Normalize the text we want to speak
    # text_norm = TTSnorm(text_to_generate)

    # # Split into sentences based on punctuation marks
    # import re
    # sentences = []
    # # Split on . ! ? , ; : ... and other Vietnamese punctuation marks
    # text_parts = re.split(r'(?<!\.)(?<!\.)\s*[.!?,;:…]+\s*(?!\.)|(?<=[\.,!?;:])\s+', text_norm)
    # # Remove empty strings and normalize
    # sentences = [TTSnorm(part.strip()) for part in text_parts if part.strip()]

    # audios_base64 = []
    # for i, sentence in enumerate(sentences):
    #     output_dir = "/root/outputs"
    #     output_path = os.path.join(output_dir, f"generated_speech_{i+1}.wav")
    #     #remove , . : ... from sentence
    #     sentence = sentence.replace(",", "").replace(".", "").replace(":", "").replace("...", "")
    #     print('sentence', sentence)
    #     # THE ACTUAL GENERATION HAPPENS HERE!
    #     tts.generate(
    #         text=sentence,
    #         output_path=output_path,
    #         nfe_step=32,               # Denoising steps. More = slower but potentially better? (Default: 32)
    #         cfg_strength=3.0,          # How strongly to stick to the reference voice style? (Default: 2.0)
    #         speed=1.0,                 # Make it talk faster or slower (Default: 1.0)
    #         cross_fade_duration=0.12,  # Smooths transitions if text is split into chunks (Default: 0.15)
    #         sway_sampling_coef=-1
    #     )
    #     print(f"Boom! Audio saved to: {output_path}")
    #     # get the audio base64
    #     with open(output_path, "rb") as f:
    #         audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    #     audios_base64.append(audio_base64)
        
    # print("\nAll done! Check your output folder.")
    # # Return all images
    # return {
    #     f"audio{i+1}": audios_base64[i] for i in range(len(audios_base64))
    # }
    
@app.function()
def process_job_inference(prompt: str, width: int = 1024, height: int = 1024, num_images: int = 6):
    return Model().t2i_inference.remote(prompt, width, height, num_images)

@app.function(timeout=3600)
def process_job_inference_framepack(prompt: str, image_url: str, 
                                    width: int = 1024, height: int = 1024, 
                                    num_images: int = 91, num_inference_steps: int = 25,
                                    guidance_scale: float = 10.0,
                                    latent_window_size: int = 9,
                                    total_second_length: float = 2.0,
                                    use_teacache: bool = True,
                                    gpu_memory_preservation: float = 6.0,
                                    mp4_crf: int = 16):
    return i2v_inference.remote(prompt, 
                                image_url, 
                                width, 
                                height, 
                                num_inference_steps, 
                                guidance_scale,
                                latent_window_size,
                                total_second_length,
                                use_teacache,
                                gpu_memory_preservation,
                                mp4_crf)

# @app.local_entrypoint()
# def main(prompt: str = "A majestic dragon soaring over snow-capped mountains",
#          compile: bool = False):
#     output_dir = Path("/tmp/flux")
#     output_dir.mkdir(exist_ok=True)

#     t0 = time.time()
#     image_bytes = Model(compile=compile).t2i_inference.remote(prompt, num_images=6)
#     print(f"Generation time: {time.time() - t0:.2f} seconds")

#     output_path = output_dir / "output.jpg"
#     output_path.write_bytes(image_bytes)
#     print(f"Saved to {output_path}")
    
    
"""
    FASTAPI ----------------------------------------
"""
web_app = fastapi.FastAPI()


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/t2i-endpoint")
async def t2i_job_endpoint(request: T2IRequest):
    process_job = modal.Function.from_name("ai-gen-flux", "process_job_inference")

    call = process_job.spawn(request.prompt, request.width, request.height, request.num_images)
    return {"call_id": call.object_id}

@web_app.post("/i2v-endpoint")
async def i2v_job_endpoint(request: I2VRequest):
    process_job = modal.Function.from_name("ai-gen-flux", "process_job_inference_framepack")
    call = process_job.spawn(request.prompt, 
                             request.image_url, 
                             request.width, 
                             request.height, 
                             request.num_images, 
                             request.num_inference_steps, 
                             request.guidance_scale,
                             request.latent_window_size,
                             request.total_second_length,
                             request.use_teacache,
                             request.gpu_memory_preservation,
                             request.mp4_crf)
    return {"call_id": call.object_id}

@web_app.get("/result/{call_id}")
async def get_job_result_endpoint(call_id: str):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except modal.exception.OutputExpiredError:
        return fastapi.responses.JSONResponse(content="", status_code=404)
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result

class CreateAudioF5TTSRequest(BaseModel):
    text_to_generate: str
    audio_url: str

@web_app.post("/create-audio-f5tts")
async def create_audio_f5tts_endpoint(request: CreateAudioF5TTSRequest):
    process_job = modal.Function.from_name("ai-gen-flux", "create_audio_f5tts")
    call = process_job.spawn(request.text_to_generate, request.audio_url)
    return {"call_id": call.object_id}
