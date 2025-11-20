#!/bin/bash

# 设置固定参数
tower="2"
root_data_dir="./dataset"
dataset="Prime_Pantry"
behaviors="Prime_Pantry_cleaned_history.csv"
news="metadata_Prime_Pantry.csv"
fre="Prime_Pantry_fre.csv"
logging_num=2
testing_num=1
bert_model_load="llm"
freeze_paras_before=0
news_attributes="title"
mode="train"
item_tower="modal_cat"
epoch=4000
num_workers=4         # 减少数据加载线程
transformer_block=2     # 减少Transformer层数
num_gpus=4
cuda_visible_devices="0,1,2,3"
l2_weight=0.1
drop_rate=0.1
batch_size=16          # 降低批次大小
embedding_dim=128       # 固定嵌入维度
lr=1e-5        # 固定学习率
mo_dnn_layers=4
dnn_layers=0
#load_ckpt_name="epoch-549_00.pt"  # 固定检查点名称
load_ckpt_name="epoch-74.pt"  # 固定检查点名称
EARLY_STOP=20
llm_embedding="Prime_Pantry_llm2vec.pt"
# 设置CUDA内存分配策略 - 移除不兼容的expandable_segments选项
# 对于PyTorch 2.0.1，使用兼容的选项
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 生成标签屏幕名称
label_screen="${item_tower}_bs${batch_size}_ed${embedding_dim}_lr${lr}_dp${drop_rate}_L2${l2_weight}"

# 生成运行命令
run_py="CUDA_VISIBLE_DEVICES='${cuda_visible_devices}' \
torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
run_amazon_Prime_Pantry_tower_2.py \
--root_data_dir ${root_data_dir} --dataset ${dataset} --behaviors ${behaviors} --news ${news} --fre ${fre} \
--mode ${mode} --tower ${tower} --item_tower ${item_tower} --load_ckpt_name ${load_ckpt_name} --label_screen ${label_screen} \
--logging_num ${logging_num} --testing_num ${testing_num} \
--l2_weight ${l2_weight} --drop_rate ${drop_rate} --batch_size ${batch_size} --lr ${lr} --embedding_dim ${embedding_dim} \
--news_attributes ${news_attributes} --bert_model_load ${bert_model_load} \
--epoch ${epoch} --freeze_paras_before ${freeze_paras_before}  --llm_embedding ${llm_embedding} \
--mo_dnn_layers ${mo_dnn_layers} --dnn_layers ${dnn_layers} --num_workers ${num_workers} --transformer_block ${transformer_block} --early_stop ${EARLY_STOP}"

# 打印并执行命令
echo "正在运行命令: ${run_py}"
eval "${run_py}"