import argparse
import os
import time
import torch
import torch.distributed as dist

from diffusers import BitsAndBytesConfig
from diffusers.utils import export_to_video

from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832) 
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--quantization", type=str, default="nf4")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    initialize_distributed()
    device = torch.cuda.current_device()
    weight_dtype = torch.bfloat16

    # Initialize model with NF4 quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["proj_out", "norm_out"]
    )

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer/",
        torch_dtype=weight_dtype,
        quantization_config=quantization_config
    )

    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=weight_dtype
    )

    pipe.enable_vae_tiling()
    
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    # Generate video
    start_time = time.perf_counter()
    
    generator = torch.Generator("cpu").manual_seed(args.seed)
    video = pipe(
        prompt="The girl is playing with her mobile phone. There are fireworks effects around her. The girl smiles happily.",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).frames

    if nccl_info.global_rank <= 0:
        os.makedirs(args.output_path, exist_ok=True)
        export_to_video(video[0], os.path.join(args.output_path, "output.mp4"), fps=24)

    generation_time = time.perf_counter() - start_time
    print(f"Video generation time: {generation_time:.2f} seconds")

if __name__ == "__main__":
    main()