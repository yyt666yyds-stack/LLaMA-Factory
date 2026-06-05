#!/bin/bash
# ============================================================
# AIFriends Qwen3-8B 训练环境快速复原脚本
# 用法: bash setup.sh
# ============================================================
set -e

cd /mnt/workspace/LLaMA-Factory

echo "=== 1. 拉取最新代码 ==="
git pull myfork main

echo "=== 2. 恢复 LLaMA-Factory 源码 ==="
if [ ! -f src/llamafactory/cli.py ]; then
    git checkout origin/main -- src/
    echo "  src/ 已恢复"
else
    echo "  src/ 已存在，跳过"
fi

echo "=== 3. 安装依赖 ==="
pip install -e . -q 2>/dev/null || pip install -e . --no-build-isolation -q

echo "=== 4. 验证 ==="
llamafactory-cli version 2>/dev/null || echo "  (版本检查跳过)"

echo ""
echo "✅ 环境就绪！启动训练："
echo "   DISABLE_VERSION_CHECK=1 WANDB_DISABLED=true llamafactory-cli train examples/train/custom/qwen3_8b_cs_qlora_v2.yaml"
