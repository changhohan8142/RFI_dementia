# -*- coding: utf-8 -*-
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,9

# chmod +x /home/hch/dementia/script/script.sh
# ./script/script.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL



ROOT_DIR="$(dirname "$0")/.."
# export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}

#lora full fine-tuning 

echo "▶ Running: retfound full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound" --ft "lora" --ft_blks "full" --enable_amp 

echo "▶ Running: retfound(dinov2) full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound_dinov2" --ft "lora" --ft_blks "full" --enable_amp 

echo "▶ Running: dinov3 full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
 "${ROOT_DIR}/trainer.py" --arch "dinov3" --ft "lora" --ft_blks "full" --enable_amp 

echo "▶ Running: dinov2 full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "dinov2" --ft "lora" --ft_blks "full" --enable_amp 

echo "▶ Running: mae full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "mae" --ft "lora" --ft_blks "full" --enable_amp 

echo "▶ Running: openclip full LoRA fine-tuning"
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "openclip" --ft "lora" --ft_blks "full" --enable_amp 



# partial fine-tuning 4 blocks
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound_dinov2" --ft "partial" --ft_blks 4 --enable_amp

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "dinov3" --ft "partial" --ft_blks 4 --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "dinov2" --ft "partial" --ft_blks 4 --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound" --ft "partial" --ft_blks 4 --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "mae" --ft "partial" --ft_blks 4 --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "openclip" --ft "partial" --ft_blks 4 --enable_amp 



# linear
torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "dinov2" --ft "linear" --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "dinov3" --ft "linear" --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound" --ft "linear" --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "retfound-dinov2" --ft "linear" --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "mae" --ft "linear" --enable_amp 

torchrun --nproc_per_node=8 \
"${ROOT_DIR}/trainer.py" --arch "openclip" --ft "linear" --enable_amp 
