#!/bin/bash

# 定义固定参数
ROOT_DATA_DIR="./dataset"
DATASET="Prime_Pantry"
BEHAVIORS="Prime_Pantry_cleaned_history.csv"
NEWS="metadata_Prime_Pantry.csv"
FRE="Prime_Pantry_fre.csv"
LOGGING_NUM=2
TESTING_NUM=1
BERT_MODEL_LOAD="llm"
FREEZE_PARAS_BEFORE=0
NEWS_ATTRIBUTES="title"
MODE="train"
TOWER="1"
ITEM_TOWER="modal_cat"
EPOCH=4000
NUM_WORKERS=8
TRANSFORMER_BLOCK=5
NUM_GPUS=4
CUDA_VISIBLE_DEVICES="0,1,2,3"

# 直接设置参数值（不需要列表和循环）
L2_WEIGHT=0.1
DROP_RATE=0.1
BATCH_SIZE=16
#BATCH_SIZE=128
EMBEDDING_DIM=128
# shellcheck disable=SC2034
GAMMA=1e-5
MO_DNN_LAYERS=4
DNN_LAYERS=0
EARLY_STOP=20
# 固定的学习率和检查点名称（针对128维度）
LR=1.5e-4
#LR=1e-4
# Tower_one就是用ID去训练获得以下文件的，所以Tower_one并不需要进行设置
LOAD_CKPT_NAME="None"
llm_embedding="Prime_Pantry_llm2vec.pt"

# 生成标签屏幕名称
LABEL_SCREEN="${ITEM_TOWER}_bs${BATCH_SIZE}_ed${EMBEDDING_DIM}_lr${LR}_dp${DROP_RATE}_L2${L2_WEIGHT}"

# 生成运行命令
RUN_COMMAND="CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' \
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
run_amazon_Prime_Pantry_tower_1.py \
--root_data_dir ${ROOT_DATA_DIR} --dataset ${DATASET} --behaviors ${BEHAVIORS} --news ${NEWS} --fre ${FRE} \
--mode ${MODE} --tower ${TOWER} --item_tower ${ITEM_TOWER} --load_ckpt_name ${LOAD_CKPT_NAME} --label_screen ${LABEL_SCREEN} \
--logging_num ${LOGGING_NUM} --testing_num ${TESTING_NUM} \
--l2_weight ${L2_WEIGHT} --drop_rate ${DROP_RATE} --batch_size ${BATCH_SIZE} --lr ${LR} --embedding_dim ${EMBEDDING_DIM} \
--news_attributes ${NEWS_ATTRIBUTES} --bert_model_load ${BERT_MODEL_LOAD} \
--epoch ${EPOCH} --freeze_paras_before ${FREEZE_PARAS_BEFORE} --llm_embedding ${llm_embedding} \
--mo_dnn_layers ${MO_DNN_LAYERS} --dnn_layers ${DNN_LAYERS} --num_workers ${NUM_WORKERS} --transformer_block ${TRANSFORMER_BLOCK} --early_stop ${EARLY_STOP}"

# 打印并执行命令
echo "正在运行命令: $RUN_COMMAND"
eval "$RUN_COMMAND"