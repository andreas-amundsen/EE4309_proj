#!/usr/bin/env bash
set -euo pipefail

# Default parameters
MODEL_INPUT=${4:-vit}
MODEL=$(echo "$MODEL_INPUT" | tr '[:upper:]' '[:lower:]')
OUTPUT=${1:-runs/${MODEL}_voc07}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-1}

echo "🚀 Starting training..."
echo "Model: $MODEL"
echo "Output directory: $OUTPUT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "📊 Dataset configuration:"
echo "- Dataset: trainval (PASCAL VOC 2007)"
echo "- Train samples: 2000 (from index 0-1999)"
echo "- Val samples: 500 (from index 2000-2499)"
echo "- Test set: reserved for final evaluation"
echo ""

python3 -m src.engine.train \
  --train-set trainval \
  --val-set trainval \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 0.002 \
  --weight-decay 1e-4 \
  --momentum 0.9 \
  --num-workers 4 \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --seed 42 \
  --train-subset-size 2000 \
  --val-subset-size 500

echo "✅ Training completed! Check results in $OUTPUT/"
