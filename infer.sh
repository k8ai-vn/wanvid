torchrun \
    --node_rank=0 \
    --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:23468 \
    --nproc_per_node=8 generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref1.png,examples/ref2.png" --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --prompt "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。" --base_seed 42

torchrun \
    --node_rank=0 \
    --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:23468 \
    --nproc_per_node=8 generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref3.png,examples/ref4.png" --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --prompt "夕阳下，一位有着小麦色肌肤、留着乌黑长发的女人穿上有着大朵立体花朵装饰、肩袖处带有飘逸纱带的红色纱裙，漫步在金色的海滩上，海风轻拂她的长发，画面唯美动人。" --base_seed 42

torchrun \
    --node_rank=0 \
    --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:23468 \
    --nproc_per_node=8 generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref5.png,examples/ref6.png,examples/ref7.png" --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --prompt "在被冰雪覆盖，周围盛开着粉色花朵，有蝴蝶飞舞，屋内透出暖黄色灯光的梦幻小屋场景下，一位头发灰白、穿着深绿色上衣的老人牵着梳着双丸子头、身着中式传统服饰、外披白色毛绒衣物的小女孩的手，缓缓前行，画面温馨宁静。" --base_seed 42

torchrun \
    --node_rank=0 \
    --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:23468 \
    --nproc_per_node=8 generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref8.png,examples/ref9.png,examples/ref10.png,examples/ref11.png" --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --prompt "一位金色长发的女人身穿棕色带波点网纱长袖、胸前系带设计的泳衣，手持一杯有橙色切片和草莓装饰、插着绿色吸管的分层鸡尾酒，坐在有着棕榈树、铺有蓝白条纹毯子和灰色垫子、摆放着躺椅的沙滩上晒日光浴的慢镜头，捕捉她享受阳光的微笑与海浪轻抚沙滩的美景。" --base_seed 42