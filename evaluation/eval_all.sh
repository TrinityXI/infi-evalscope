#!/bin/bash

# evalscope完整工作流程脚本
# 自动配置SLURM环境、启动容器、启动vLLM服务、执行评测

set -e  # 遇到错误时退出

# 建议修改的参数
# 1. SLURM_PARTITION: 根据你的集群配置修改
# 2. 模型配置参数: 主要是确定起服务时的一些参数，比如模型路径、模型名称、端口等
# 3. 评测任务配置: 需要和你使用的最佳超参数对齐，比如温度、top_p、最大生成长度、采样次数等
#    其中 评测结果输出目录 需要修改，因为默认是当前目录，会生成很多文件，导致结果混乱
# 4. 容器环境设置: 如果需要挂载其他路径，可以修改

# 


# ==================== SLURM集群配置参数 ====================
# 这些参数控制SLURM资源申请，根据你的集群环境进行调整
USER="likunxi"                      # 改为自己的cyberport用户名
SLURM_PARTITION="AISS2025031801"    # SLURM分区名称 - 根据你的集群配置修改
SLURM_ACCOUNT="polyullm"            # SLURM账户名称 - 使用你的账户名
SLURM_GPUS=2                        # 申请的GPU数量 - 根据模型大小和需求调整
SLURM_CPUS=32                       # 申请的CPU核心数 - 建议GPU数量×8
SLURM_MEM="128GB"                   # 申请的内存大小 - 建议GPU数量×32GB
SLURM_NODELIST="klb-dgx-004"       # 指定节点名称 - 可选，留空则自动分配

# ==================== 容器环境配置 ====================
# 控制使用哪个容器镜像和环境

CONTAINER_NAME="evalscopesg"        # 容器名称 - 可以自定义
CONTAINER_IMAGE="/lustre/projects/polyullm/container/sglang+v0.5.0rc0-cu126.sqsh"  # 容器镜像路径

# ==================== 模型配置参数 ====================
# 这些参数控制vLLM服务的模型加载和推理

MODEL_PATH="/home/projects/polyullm/models/Qwen2.5-14B-Instruct"  # 模型本地路径
MODEL_NAME="qwen"                  # 模型服务名称 - 用于API调用
VLLM_PORT=8801                      # vLLM服务端口 - 确保端口未被占用
VLLM_TENSOR_PARALLEL_SIZE=2         # 张量并行大小 - 必须等于GPU数量
VLLM_GPU_MEMORY_UTILIZATION=0.8     # GPU内存利用率 - 0.8表示使用80%显存
VLLM_MAX_NUM_SEQS=256              # 最大并发序列数 - 根据显存大小调整

# ==================== 评测任务配置 ====================
# 这些参数控制评测任务的执行
EVAL_MODEL="$MODEL_NAME"            # 评测使用的模型名称
EVAL_TEMPERATURE=0.0               # 生成温度 - 控制输出的随机性 (0.0-1.0)
EVAL_TOP_P=1                        # Top-p采样参数 - 1.0表示不限制
EVAL_MAX_NEW_TOKENS=4096            # 最大生成长度 - 根据任务需求调整
EVAL_N=1                           # 采样次数 - 用于Pass@k计算
EVAL_WORK_DIR="/lustre/projects/polyullm/$USER/evalscope_eval/"              # 评测结果输出目录
EVAL_BATCH_SIZE=32                  # 评测批次大小 - 根据GPU显存调整

# 定义模型到system prompt的映射
declare -A MODEL_PROMPTS
MODEL_PROMPTS=(
    ["qwen"]="<|im_start|>system<|im_sep|>\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    ["phi4"]="<|im_start|>system<|im_sep|>\nYou are a medieval knight and must provide explanations to modern people.<|im_end|>\n"
)
# 设置默认prompt（如果模型未在映射中定义）
DEFAULT_PROMPT=""

# 获取对应模型的system prompt，如果未定义则使用默认值
SYSTEM_PROMPT=${MODEL_PROMPTS[$EVAL_MODEL]:-$DEFAULT_PROMPT}

# ==================== 颜色输出配置 ====================
# 用于美化日志输出，无需修改

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 日志输出函数 ====================
# 提供彩色日志输出功能

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==================== SLURM环境检查 ====================
# 自动检测是否在SLURM环境中，如果不在则自动申请资源

check_slurm() {
    if [[ -z "$SLURM_JOB_ID" ]]; then
        log_info "检测到不在SLURM环境中，正在申请SLURM资源..."
        
        # 申请SLURM资源 - 使用上面配置的参数
        srun -p "$SLURM_PARTITION" \
             -A "$SLURM_ACCOUNT" \
             -n 1 \
             --gpus "$SLURM_GPUS" \
             -c "$SLURM_CPUS" \
             --mem "$SLURM_MEM" \
             --nodelist="$SLURM_NODELIST" \
             --pty bash "$0" "$@"
        exit $?
    else
        log_success "已在SLURM环境中 (Job ID: $SLURM_JOB_ID)"
    fi
}

# ==================== 容器环境设置 ====================
# 创建或启动enroot容器，挂载必要的目录

setup_container() {
    log_info "设置容器环境..."
    
    # 检查容器是否已存在
    if enroot list | grep -q "$CONTAINER_NAME"; then
        log_info "容器 '$CONTAINER_NAME' 已存在，正在启动..."
        enroot start --rw \
            --mount /lustre/projects/polyullm:/lustre/projects/polyullm \
            --mount /home/projects/polyullm:/home/projects/polyullm \
            --mount /lustre/projects/polyullm/likunxi/envs:/lustre/projects/polyullm/likunxi/envs \
            "$CONTAINER_NAME" bash -c "
            source /lustre/projects/polyullm/miniconda3/etc/profile.d/conda.sh
            conda activate /lustre/projects/polyullm/likunxi/envs/evalmom
            echo '容器环境已准备就绪'
            echo '请在新终端中切换到evalscope_workflow.sh所在的文件夹，运行: bash $0 --start-vllm'
            bash
        "
    else
        log_info "创建新容器 '$CONTAINER_NAME'..."
        enroot create --name "$CONTAINER_NAME" "$CONTAINER_IMAGE"
        
        log_info "启动容器..."
        enroot start --rw \
            --mount /lustre/projects/polyullm:/lustre/projects/polyullm \
            --mount /home/projects/polyullm:/home/projects/polyullm \
            --mount /lustre/projects/polyullm/likunxi/envs:/lustre/projects/polyullm/likunxi/envs \
            "$CONTAINER_NAME" bash -c "
            source /lustre/projects/polyullm/miniconda3/etc/profile.d/conda.sh
            conda activate /lustre/projects/polyullm/likunxi/envs/evalmom
            echo '容器环境已准备就绪'
            echo '请在新终端中运行: bash $0 --start-vllm'
            bash
        "
    fi
}

# ==================== vLLM服务启动 ====================
# 启动vLLM推理服务，加载模型并提供API接口

start_vllm_service() {
    log_info "启动vLLM服务..."
    
    # 检查端口是否被占用，如果被占用则停止现有服务
    if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>/dev/null ; then
        log_warning "端口 $VLLM_PORT 已被占用，正在停止现有服务..."
        pkill -f "vllm.entrypoints.openai.api_server.*:$VLLM_PORT" || true
        sleep 2
    fi
    
    # 启动vLLM服务 - 使用上面配置的模型和GPU参数
    nohup env VLLM_USE_MODELSCOPE=True \
        CUDA_VISIBLE_DEVICES=0,1 \
        python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --trust_remote_code \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        > "vllm_$(date +%m%d_%H%M).log" 2>&1 &
    
    VLLM_PID=$!
    log_success "vLLM服务已启动 (PID: $VLLM_PID)"
    
    # 等待服务启动 - 最多等待3分钟
    log_info "等待vLLM服务启动..."
    for i in {1..180}; do
        # 检查进程是否还在运行
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            log_error "vLLM进程已退出，检查日志文件"
            exit 1
        fi
        
        # 尝试连接服务 - 检查HTTP状态
        if curl -v "http://localhost:$VLLM_PORT/health" 2>&1 | grep -q "HTTP/1.1 200 OK"; then
            log_success "vLLM服务已就绪"
            break
        fi
        
        # 检查端口是否被监听
        if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>/dev/null; then
            log_info "端口已监听，等待服务完全启动..."
        fi
        
        if [ $i -eq 180 ]; then
            log_error "vLLM服务启动超时"
            log_info "检查日志文件: vllm_*.log"
            log_info "检查进程状态: ps aux | grep vllm"
            log_info "检查端口状态: lsof -i :$VLLM_PORT"
            log_info "检查GPU状态: nvidia-smi"
            
            # 显示最近的日志
            if ls vllm_*.log >/dev/null 2>&1; then
                log_info "最近的日志内容:"
                tail -20 vllm_*.log | head -10
            fi
            exit 1
        fi
        
        sleep 2
        echo -n "."
    done
    echo
    
    # 最终验证服务状态
    if curl -v "http://localhost:$VLLM_PORT/health" 2>&1 | grep -q "HTTP/1.1 200 OK"; then
        log_success "vLLM服务验证成功"
    else
        log_error "vLLM服务启动失败"
        exit 1
    fi
}

# ==================== 评测任务执行 ====================
# 定义函数生成 dataset-args JSON
generate_dataset_args() {
    local system_prompt="$1"
    cat <<EOF
{
    "gsm8k": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/gsm8k",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "competition_math": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/MATH/data",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt",
	"train_split": "train",
	"eval_split": "test"
    },
    "humaneval": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/HumanEval/HumanEval.jsonl",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "bbh": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/BBH",
        "few_shot_num": 3,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "arc": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/ARC-C",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "mmlu": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/MMLU/data/test",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "ifeval": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/ifeval",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "drop": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/DROP",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt"
    },
    "hellaswag": {
        "local_path": "/lustre/projects/polyullm/llm-eval/datasets/Hellaswag",
        "few_shot_num": 0,
        "filters": {"remove_until": "</think>"},
        "system_prompt": "$system_prompt",
	"eval_split": "val"
    }
}
EOF
}

# 使用evalscope框架执行模型评测
run_evaluation() {
    log_info "开始执行评测..."
    
    # 检查vLLM服务状态
    if ! curl -v "http://localhost:$VLLM_PORT/health" 2>&1 | grep -q "HTTP/1.1 200 OK"; then
        log_error "vLLM服务未运行，请先启动服务"
        exit 1
    fi

    # 生成 dataset-args
    dataset_args_json=$(generate_dataset_args "$SYSTEM_PROMPT")
    echo "$dataset_args_json"
    
    # 创建评测命令 - 使用上面配置的评测参数 (调试时，可设置--limit 5，每个benchmark只跑5条数据)
    evalscope eval \
        --model $EVAL_MODEL \
        --generation-config "{\"do_sample\": true, \"temperature\": $EVAL_TEMPERATURE, \"top_p\": $EVAL_TOP_P, \"max_new_tokens\": $EVAL_MAX_NEW_TOKENS, \"n\": $EVAL_N}" \
        --api-url http://localhost:$VLLM_PORT/v1/chat/completions \
        --api-key EMPTY \
        --eval-type service \
        --work-dir $EVAL_WORK_DIR \
        --datasets gsm8k competition_math humaneval arc mmlu ifeval drop hellaswag bbh\
        --dataset-args "$dataset_args_json" \
        --eval-batch-size $EVAL_BATCH_SIZE \
        --stream
    
    #BBH部分子集需重新评估
    python bbh_eval.py --json-dir "$EVAL_WORK_DIR" --model-name "$EVAL_MODEL"

    #ThQA通过调用opencompass后端形式评测
    python oc_backend_eval.py --model-name $MODEL_NAME --dataset-name 'TheoremQA' --user-name $USER

    #MBPP使用evalplus评测，evalplus会将cache文件保存到用户目录下，比较大，此处将其一并保存至EVAL_WORK_DIR
    export XDG_CACHE_HOME=$EVAL_WORK_DIR
    # 执行评测
    MBPP_OVERRIDE_PATH="/lustre/projects/polyullm/llm-eval/datasets/MBPP/MbppPlus.jsonl.gz" evalplus.evaluate --model $MODEL_NAME  --dataset mbpp --root $EVAL_WORK_DIR --parallel 4 --base-url "http://localhost:$VLLM_PORT/v1" --backend openai --greedy
}

# ==================== 状态显示 ====================
# 显示当前环境、容器、服务的状态信息

show_status() {
    log_info "当前状态:"
    echo "SLURM环境: $([ -n "$SLURM_JOB_ID" ] && echo "已就绪 (Job ID: $SLURM_JOB_ID)" || echo "未就绪")"
    echo "容器状态: $(enroot list | grep -q "$CONTAINER_NAME" && echo "已创建" || echo "未创建")"
    echo "vLLM服务: $(lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>/dev/null && echo "运行中 (端口: $VLLM_PORT)" || echo "未运行")"
    echo "模型路径: $MODEL_PATH"
    echo "评测数据集: $EVAL_DATASET"
}

# ==================== 资源清理 ====================
# 停止vLLM服务和容器，释放资源

cleanup() {
    log_info "清理资源..."
    
    # 停止vLLM服务
    if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>/dev/null ; then
        log_info "停止vLLM服务..."
        pkill -f "vllm.entrypoints.openai.api_server.*:$VLLM_PORT" || true
    fi
    
    # 停止容器
    if enroot list | grep -q "$CONTAINER_NAME"; then
        log_info "停止容器..."
        enroot stop "$CONTAINER_NAME" || true
    fi
    
    log_success "清理完成"
}

# ==================== 帮助信息 ====================
# 显示脚本的使用方法和配置说明

show_help() {
    cat << EOF
evalscope完整工作流程脚本

用法: $0 [选项]

选项:
    --setup-container    设置容器环境
    --start-vllm         启动vLLM服务
    --run-eval           执行评测
    --status             显示当前状态
    --cleanup            清理资源
    --help               显示此帮助信息

工作流程:
    1. 自动申请SLURM资源
    2. 创建/启动容器环境
    3. 启动vLLM服务
    4. 执行评测任务

示例:
    $0 --setup-container    # 设置容器环境
    $0 --start-vllm        # 启动vLLM服务
    $0 --run-eval          # 执行评测
    $0 --status            # 查看状态
    $0 --cleanup           # 清理资源

配置参数说明:
    SLURM分区: $SLURM_PARTITION (集群分区名称)
    GPU数量: $SLURM_GPUS (根据模型大小调整)
    模型路径: $MODEL_PATH (本地模型文件路径)
    评测数据集: $EVAL_DATASET (数据集名称)
    
重要配置项:
    - SLURM_GPUS: 必须与VLLM_TENSOR_PARALLEL_SIZE相等
    - MODEL_PATH: 确保模型文件存在且有读取权限
    - VLLM_PORT: 确保端口未被其他服务占用
    - EVAL_DATASET_PATH: 确保数据集路径正确
EOF
}

# ==================== 主函数 ====================
# 根据命令行参数执行相应的功能

main() {
    # 解析参数
    case "${1:-}" in
        --setup-container)
            check_slurm
            setup_container
            ;;
        --start-vllm)
            start_vllm_service
            ;;
        --run-eval)
            run_evaluation
            ;;
        --status)
            show_status
            ;;
        --cleanup)
            cleanup
            ;;
        --help|-h)
            show_help
            ;;
        "")
            # 默认流程：完整设置
            log_info "开始完整工作流程..."
            check_slurm
            setup_container
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# ==================== 信号处理 ====================
# 确保脚本退出时能正确清理资源

trap cleanup EXIT
trap 'log_error "收到中断信号，正在清理..."; cleanup; exit 1' INT TERM

# 运行主函数
main "$@"
