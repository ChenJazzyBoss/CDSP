#!/bin/bash

# 参数设置
tower="3"
root_data_dir="./dataset"
dataset="Prime_Pantry"
behaviors="Prime_Pantry_cleaned_history.csv"
news="metadata_Prime_Pantry.csv"
fre="Prime_Pantry_fre.csv"
item_tower="modal_cat"
#load_ckpt_name="epoch-753_DAE_V9_42_residual.pt"
load_ckpt_name="epoch-187-7-denoise_7.pt"
#batch_size=128
batch_size=32
embedding_dim=128
lr=1e-5
l2_weight=0.1
drop_rate=0.1
freeze_paras_before=0
num_gpus=4
cuda_visible_devices="0,1,2,3"
num_workers=8          # 保持与之前一致的线程数
transformer_block=5    # 恢复原始层数（根据需要调整）
gamma2=1e-4
epoch=4000
logging_num=2
testing_num=1
bert_model_load="llm"
news_attributes="title"
mo_dnn_layers=4
dnn_layers=0
EARLY_STOP=50
llm_embedding="Prime_Pantry_llm2vec.pt"
# 生成标签
label_screen="${item_tower}_bs${batch_size}_ed${embedding_dim}_lr${lr}_dp${drop_rate}_L2${l2_weight}_Flr${freeze_paras_before}"

# 运行命令
# shellcheck disable=SC2089
run_cmd="CUDA_VISIBLE_DEVICES='${cuda_visible_devices}' \
torchrun --nproc_per_node=${num_gpus} --master_port 12345 \
run_amazon_Prime_Pantry_tower_3.py \
--root_data_dir ${root_data_dir} --dataset ${dataset} --behaviors ${behaviors} --news ${news} --fre ${fre} \
--mode train --tower ${tower} --item_tower ${item_tower} --load_ckpt_name ${load_ckpt_name} --label_screen ${label_screen} \
--logging_num ${logging_num} --testing_num ${testing_num} --l2_weight ${l2_weight} --drop_rate ${drop_rate} \
--batch_size ${batch_size} --lr ${lr} --embedding_dim ${embedding_dim} --news_attributes ${news_attributes} \
--bert_model_load ${bert_model_load} --epoch ${epoch} --freeze_paras_before ${freeze_paras_before} \
--mo_dnn_layers ${mo_dnn_layers} --dnn_layers ${dnn_layers} --gamma2 ${gamma2}  --llm_embedding ${llm_embedding} \
--num_workers ${num_workers} --transformer_block ${transformer_block} --early_stop ${EARLY_STOP}"

# 执行
echo "正在运行命令:"
# shellcheck disable=SC2090
echo "$run_cmd" | tr ' ' '\n'  # 分行显示命令（可选）
eval "$run_cmd"