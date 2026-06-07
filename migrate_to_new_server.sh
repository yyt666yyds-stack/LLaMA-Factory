#!/bin/bash
# ============================================================
# AIFriends Qwen3-8B 新服务器一键部署脚本
# 用法: bash migrate_to_new_server.sh
# ============================================================
set -e

WORKSPACE="/mnt/workspace"
cd $WORKSPACE

echo "============================================"
echo " Qwen3-8B 训练环境部署脚本"
echo "============================================"

# ====== 1. Clone 代码仓库 ======
echo ""
echo "=== [1/5] Clone LLaMA-Factory Fork ==="
if [ ! -d "$WORKSPACE/LLaMA-Factory" ]; then
    git clone git@github.com:yyt666yyds-stack/LLaMA-Factory.git
    cd LLaMA-Factory
    git remote add origin https://github.com/hiyouga/LLaMA-Factory.git
else
    cd $WORKSPACE/LLaMA-Factory
    git pull myfork main
fi

# ====== 2. 安装 LLaMA-Factory ======
echo ""
echo "=== [2/5] 安装 LLaMA-Factory 依赖 ==="
pip install -e . -q 2>/dev/null || pip install -e . --no-build-isolation -q

# ====== 3. 下载基础模型 ======
echo ""
echo "=== [3/5] 下载基础模型 Qwen3-8B-Instruct ==="
mkdir -p $WORKSPACE/models
if [ ! -d "$WORKSPACE/models/Qwen/Qwen3-8B-Instruct" ]; then
    modelscope download --model Qwen/Qwen3-8B-Instruct --local_dir $WORKSPACE/models/Qwen/Qwen3-8B-Instruct
    echo "  基础模型下载完成"
else
    echo "  基础模型已存在，跳过"
fi

# ====== 4. 下载已训练的 LoRA (可选) ======
echo ""
echo "=== [4/5] 下载已训练的 LoRA Adapter (可选) ==="
echo "  V1 (客服对话): modelscope download --model YveanMask/qwen3-8b-customer-service-qlora"
echo "  V2 (客服+工具): modelscope download --model YveanMask/qwen3-8b-customer-service-qlora-v2"
echo "  跳过自动下载，需要时手动执行"

# ====== 5. 验证环境 ======
echo ""
echo "=== [5/5] 验证环境 ==="
llamafactory-cli version
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)')
"

echo ""
echo "============================================"
echo " ✅ 环境部署完成！"
echo "============================================"
echo ""
echo " 启动训练："
echo "   cd $WORKSPACE/LLaMA-Factory"
echo "   DISABLE_VERSION_CHECK=1 WANDB_DISABLED=true llamafactory-cli train examples/train/custom/qwen3_8b_cs_qlora_v2.yaml"
echo ""
echo " 从 checkpoint 恢复训练："
echo "   llamafactory-cli train examples/train/custom/qwen3_8b_cs_qlora_v2.yaml --resume_from_checkpoint saves/qwen3-8b-customer-service-v2/checkpoint-1773"
echo ""
