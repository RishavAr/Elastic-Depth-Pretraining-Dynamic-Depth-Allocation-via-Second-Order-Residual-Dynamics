#!/bin/bash
# Run complete EDP experiment pipeline

# Exit on error
set -e

echo "=============================================="
echo "Elastic-Depth Pretraining - Full Pipeline"
echo "=============================================="

# Configuration
OUTPUT_DIR="./outputs/full_run_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
BATCH_SIZE=32
SEQ_LEN=256

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_TEST="--quick_test"
            echo "Running in QUICK TEST mode"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pipeline
python -m experiments.run_full_pipeline \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    $QUICK_TEST

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
