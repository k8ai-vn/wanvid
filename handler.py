import os
import time
import argparse
import torch

from fastvideo import VideoGenerator, SamplingParam

def main(args):
    # Set the attention backend
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    
    # Print GPU info
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GiB")
        torch.cuda.reset_max_memory_allocated(device)
    
    print(f"Loading model: {args.model_path}")
    start_time = time.perf_counter()
    
    # Initialize generator with quantization if specified
    gen = VideoGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        use_cpu_offload=args.cpu_offload,
        quantization=args.quantization
    )
    
    load_time = time.perf_counter() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    
    if torch.cuda.is_available():
        print(f"VRAM used for model loading: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        torch.cuda.reset_max_memory_allocated(device)
    
    # Prepare output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process either a single prompt or multiple prompts from a file
    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        prompts = [args.prompt]
    
    for i, prompt in enumerate(prompts):
        gen_start_time = time.perf_counter()
        
        # Configure sampling parameters
        params = SamplingParam.from_pretrained(
            model_path=args.model_path,
        )
        
        # Set TEACache parameters if enabled
        if args.enable_teacache:
            params.teacache_params.teacache_thresh = args.teacache_thresh
        
        # Generate video
        output_file = f"{args.output_path}/video_{i+1:03d}.mp4"
        print(f"Generating video {i+1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        
        gen.generate_video(
            prompt=prompt,
            sampling_param=params,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            enable_teacache=args.enable_teacache,
            seed=args.seed,
            output_path=output_file
        )
        
        generation_time = time.perf_counter() - gen_start_time
        print(f"Video {i+1} generation time: {generation_time:.2f} seconds")
        
        if torch.cuda.is_available():
            print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
            torch.cuda.reset_max_memory_allocated(device)
    
    total_time = time.perf_counter() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastVideo Generator")
    parser.add_argument("--model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Path or name of the pretrained model")
    parser.add_argument("--prompt", type=str, 
                        default="The girl is playing with her mobile phone. There are fireworks effects around her. The girl smiles happily.",
                        help="Text prompt for video generation")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to a file containing prompts (one per line)")
    parser.add_argument("--output_path", type=str, default="example_outputs",
                        help="Directory to save generated videos")
    parser.add_argument("--height", type=int, default=480,
                        help="Height of generated video")
    parser.add_argument("--width", type=int, default=832,
                        help="Width of generated video")
    parser.add_argument("--num_frames", type=int, default=61,
                        help="Number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=6,
                        help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable CPU offloading for memory efficiency")
    parser.add_argument("--enable_teacache", action="store_true", default=True,
                        help="Enable TEACache for faster generation")
    parser.add_argument("--teacache_thresh", type=float, default=0.08,
                        help="TEACache threshold value")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "nf4", "int8"],
                        help="Quantization method for reduced memory usage")
    
    args = parser.parse_args()
    main(args)