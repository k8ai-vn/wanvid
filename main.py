import time
from io import BytesIO
from pathlib import Path
import os
import modal
import fastapi
import base64

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"
NUM_INFERENCE_STEPS = 1
cuda_dev_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])

cuda_dev_image = cuda_dev_image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1",
     "XFORMERS_ENABLE_TRITON": "1",
     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # turn on faster downloads from HF
)

class ModelConfig:
    # model_name: str = "black-forest-labs/FLUX.1-dev"
    model_name: str = "black-forest-labs/FLUX.1-schnell"
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
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
        "fastapi[standard]",
        "optimum-quanto",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)
flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

# Storing data created by our app with modal.Volume
volume = modal.Volume.from_name(
    "flux-text-to-images-data", 
    create_if_missing=True
)

MODEL_DIR = "/models"

app = modal.App("flux", image=flux_image)

with flux_image.imports():
    import diffusers
    import torch

@app.function(volumes={MODEL_DIR: volume}, secrets=[modal.Secret.from_name("huggingface")])
def download_model():
    from huggingface_hub import snapshot_download
    from huggingface_hub import login
    login(token=os.environ["hg_token"])
    snapshot_download(
        ModelConfig.model_name,
        local_dir=MODEL_DIR,
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

@app.function(gpu="any")
def check_nvidia_smi():
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    print(output)
    return output

@app.cls(gpu="L4",
        #  gpu="L40S", 
         timeout=3600, secrets=[modal.Secret.from_name("huggingface")], 
         max_containers=10,
         volumes={MODEL_DIR: volume,
            "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
            "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
            "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
            "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
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
        self.pipe = optimize(self.pipe, compile=self.compile)

        
    @modal.method()
    def t2i_inference(self, prompt: str) -> bytes:
        print("Generating image...")
        generator = torch.Generator().manual_seed(42)
        image = self.pipe(
            prompt,
            width=1024,
            height=1024,
            output_type="pil",
            generator=generator,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=3.5,
            max_sequence_length=512,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        # output the image as a base64 string
        return {"image": base64.b64encode(byte_stream.getvalue()).decode("utf-8")}
    
@app.function()
def process_job_inference(prompt: str):
    return Model().t2i_inference.remote(prompt)

@app.local_entrypoint()
def main(prompt: str = "A majestic dragon soaring over snow-capped mountains",
         compile: bool = False):
    output_dir = Path("/tmp/flux")
    output_dir.mkdir(exist_ok=True)

    t0 = time.time()
    image_bytes = Model(compile=compile).t2i_inference.remote(prompt)
    print(f"Generation time: {time.time() - t0:.2f} seconds")

    output_path = output_dir / "output.jpg"
    output_path.write_bytes(image_bytes)
    print(f"Saved to {output_path}")
    
    
"""
    FASTAPI ----------------------------------------
"""
web_app = fastapi.FastAPI()


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/t2i-endpoint")
async def t2i_job_endpoint(data):
    process_job = modal.Function.from_name("flux", "process_job_inference")

    call = process_job.spawn(data)
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