import os
import time

from fastvideo import VideoGenerator, SamplingParam

def main():
    # set the attention backend 
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    start_time = time.perf_counter()
    gen = VideoGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
        use_cpu_offload=False,
    )
    load_time = time.perf_counter() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")

    gen_start_time = time.perf_counter()

    params = SamplingParam.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    # this controls the threshold for the tea cache
    params.teacache_params.teacache_thresh = 0.08
    gen.generate_video(
        prompt=
        "The girl is playing with her mobile phone. There are fireworks effects around her. The girl smiles happily.",
        sampling_param=params,
        height=480,
        width=832,
        num_frames=61,  # 85 ,77 
        num_inference_steps=6,
        enable_teacache=True,
        seed=1024,
        output_path="example_outputs/")
    
    generation_time = time.perf_counter() - gen_start_time
    print(f"Video generation time: {generation_time:.2f} seconds")

    total_time = time.perf_counter() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()