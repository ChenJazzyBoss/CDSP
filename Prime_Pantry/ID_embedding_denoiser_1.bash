#!/bin/bash

# =========================================================
# 嵌入去噪脚本 - 用于SRSRec模型的嵌入增强
# =========================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 默认参数
INPUT_PATH="/home/admin1/code/BitBlitz/PAD/checkpoint/Prime_Pantry/Tower_2_128/epoch-131.pt"
OUTPUT_PATH="/home/admin1/code/BitBlitz/PAD/checkpoint/Prime_Pantry/Tower_2_128/epoch-131_7_DA.pt"
EMBEDDING_DIM=128
TRAIN_EPOCHS=100
DENOISE_STEPS=50
BATCH_SIZE=16
GPU_ID=3
LOG_DIR="/home/admin1/code/BitBlitz/PAD/logs/denoise"
SCRIPT_PATH="/home/admin1/code/BitBlitz/PAD/model/denoise_embedding_7_DA.py"

#SCRIPT_PATH="/home/admin1/code/BitBlitz/PAD/model/denoise_embedding_7_DA.py"
# 创建日志目录
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/denoise_$(date +%Y%m%d_%H%M%S).log"

# 帮助函数
function show_help {
    echo -e "${BLUE}使用方法:${NC} $0 [选项]"
    echo -e "${BLUE}选项:${NC}"
    echo -e "  ${YELLOW}--input-path${NC}     输入嵌入文件路径 (默认: $INPUT_PATH)"
    echo -e "  ${YELLOW}--output-path${NC}    输出嵌入文件路径 (默认: $OUTPUT_PATH)"
    echo -e "  ${YELLOW}--embedding-dim${NC}  嵌入维度 (默认: $EMBEDDING_DIM)"
    echo -e "  ${YELLOW}--train-epochs${NC}   训练轮数 (默认: $TRAIN_EPOCHS)"
    echo -e "  ${YELLOW}--denoise-steps${NC}  去噪步数 (默认: $DENOISE_STEPS)"
    echo -e "  ${YELLOW}--batch-size${NC}     批次大小 (默认: $BATCH_SIZE)"
    echo -e "  ${YELLOW}--gpu-id${NC}         GPU编号 (默认: $GPU_ID)"
    echo -e "  ${YELLOW}--script-path${NC}    Python脚本路径 (默认: $SCRIPT_PATH)"
    echo -e "  ${YELLOW}--help${NC}           显示此帮助信息"
}

# 日志函数
function log {
    # shellcheck disable=SC2155
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo -e "$message" >> "$LOG_FILE"
}

# 错误处理函数
function handle_error {
    log "${RED}错误:${NC} $1"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --embedding-dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --train-epochs)
            TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --denoise-steps)
            DENOISE_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --script-path)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log "${RED}未知参数:${NC} $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查输入文件是否存在
if [ ! -f "$INPUT_PATH" ]; then
    handle_error "输入文件不存在: $INPUT_PATH"
fi

# 检查输出文件路径
output_dir=$(dirname "$OUTPUT_PATH")
if [ ! -d "$output_dir" ]; then
    log "${YELLOW}警告:${NC} 输出目录不存在，将尝试创建"
    mkdir -p "$output_dir" || handle_error "无法创建输出目录: $output_dir"
    log "${GREEN}成功创建目录:${NC} $output_dir"
fi

# 检查Python脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    handle_error "Python脚本不存在: $SCRIPT_PATH"
fi

# 检查GPU是否可用
if [ "$GPU_ID" -ge 0 ]; then
    GPU_AVAILABLE=$(nvidia-smi -L 2>/dev/null | grep "GPU $GPU_ID" | wc -l)
    if [ "$GPU_AVAILABLE" -eq 0 ]; then
        log "${YELLOW}警告:${NC} GPU $GPU_ID 不可用，将使用CPU"
        GPU_ID=-1
    fi
fi

# 打印配置信息
log "${BLUE}===== 嵌入去噪配置 ====${NC}"
log "输入路径: ${GREEN}$INPUT_PATH${NC}"
log "输出路径: ${GREEN}$OUTPUT_PATH${NC}"
log "嵌入维度: ${GREEN}$EMBEDDING_DIM${NC}"
log "训练轮数: ${GREEN}$TRAIN_EPOCHS${NC}"
log "去噪步数: ${GREEN}$DENOISE_STEPS${NC}"
log "批次大小: ${GREEN}$BATCH_SIZE${NC}"
log "GPU编号: ${GREEN}$GPU_ID${NC}"
log "日志文件: ${GREEN}$LOG_FILE${NC}"
log "${BLUE}=====================${NC}"

# 设置CUDA可见设备
if [ "$GPU_ID" -ge 0 ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    log "${GREEN}使用GPU $GPU_ID${NC}"
else
    export CUDA_VISIBLE_DEVICES=""
    log "${YELLOW}使用CPU${NC}"
fi

# 执行Python脚本并捕获输出
log "${BLUE}开始执行嵌入去噪...${NC}"

# 正确构建并执行Python命令
log "执行命令: python $SCRIPT_PATH --input_path $INPUT_PATH --output_path $OUTPUT_PATH --train_epochs $TRAIN_EPOCHS --denoise_steps $DENOISE_STEPS --batch_size $BATCH_SIZE --embedding_dim $EMBEDDING_DIM"

python_output=$(python "$SCRIPT_PATH" \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --train_epochs "$TRAIN_EPOCHS" \
    --denoise_steps "$DENOISE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --embedding_dim "$EMBEDDING_DIM" 2>&1)

# 将Python输出保存到日志文件
echo "$python_output" >> "$LOG_FILE"

# 检查Python脚本是否成功完成
if echo "$python_output" | grep -q "Processing completed!"; then
    log "${GREEN}嵌入去噪完成!${NC}"

    # 验证输出文件是否存在
    if [ -f "$OUTPUT_PATH" ]; then
        log "结果保存在: ${GREEN}$OUTPUT_PATH${NC}"

        # 显示输出文件信息
        file_size=$(du -h "$OUTPUT_PATH" | awk '{print $1}')
        log "输出文件大小: ${GREEN}$file_size${NC}"

        # 显示备份文件信息
        backup_file=$(echo "$python_output" | grep "Created backup" | awk '{print $NF}')
        if [ -n "$backup_file" ] && [ -f "$backup_file" ]; then
            backup_size=$(du -h "$backup_file" | awk '{print $1}')
            log "备份文件: ${GREEN}$backup_file${NC} (大小: $backup_size)"
        fi
    else
        log "${RED}错误:${NC} 输出文件不存在: $OUTPUT_PATH"
        log "${RED}Python脚本输出:${NC}"
        # shellcheck disable=SC2001
        echo "$python_output" | sed 's/^/  /'  # 缩进显示
        handle_error "输出文件验证失败"
    fi
else
    log "${RED}错误:${NC} 嵌入去噪未成功完成"
    log "${RED}Python脚本输出:${NC}"
    echo "$python_output" | sed 's/^/  /'  # 缩进显示

    # 检查是否有错误信息
    error_msg=$(echo "$python_output" | grep -i "error\|exception" | head -n 1)
    if [ -n "$error_msg" ]; then
        log "${RED}错误详情:${NC} $error_msg"
    fi

    handle_error "Python脚本执行失败"
fi