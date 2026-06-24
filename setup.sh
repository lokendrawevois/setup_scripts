#!/bin/bash
set -euo pipefail

# =============================================================================
# Benchmark log setup
# =============================================================================
BENCHMARK_LOG="benchmark_$(date '+%Y%m%d_%H%M%S').log"
# If you prefer a fixed name each run, use: BENCHMARK_LOG="benchmark.log"

# Function: write a line to the benchmark log only
log_benchmark() {
    # Format: section_name,start_time,end_time,duration_seconds
    echo "$1,$2,$3,$4" >> "$BENCHMARK_LOG"
}

# Write a header (optional)
echo "section,start,end,duration_sec" > "$BENCHMARK_LOG"

# =============================================================================
# Environment Checks
# =============================================================================
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "Please export it before running the script:"
    echo "export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set."
    echo "Please export it before running the script:"
    echo "export GOOGLE_APPLICATION_CREDENTIALS='your-api-key'"
    exit 1
fi

if [ -z "${YOLO_MODEL_PATH:-}" ]; then
    echo "ERROR: YOLO_MODEL_PATH is not set."
    echo "Please export it before running the script:"
    echo "export YOLO_MODEL_PATH='/absolute/path/to/best.pt'"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') GEMINI_API_KEY detected."

# =============================================================================
# SECTION 1: Clone repository and download models
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 1 START: Clone repo and download models"
echo "Start time: $START_TIME"
echo "====================================================="

git clone https://github.com/lokendrawevois/pipeline.git

echo "SAM3 download section"
if [ ! -f pipeline/sam3.pt ]; then
    wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt
else
    echo "sam3.pt already exists, skipping download."
fi
echo "SAM3 download section end"

if [ ! -f pipeline/yolo26n.pt ]; then
    wget https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt -O pipeline/yolo26n.pt
else
    echo "yolo26n.pt already exists, skipping download."
fi

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_1" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 1 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# SECTION 2: Setup Python environment and install dependencies
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 2 START: Setup Python environment"
echo "Start time: $START_TIME"
echo "====================================================="

cd pipeline

uv venv --python 3.9.6
source .venv/bin/activate

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_2" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 2 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# SECTION 3: Download training data
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 3 START: Download training data"
echo "Start time: $START_TIME"
echo "====================================================="

python download_training_data.py

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_3" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 3 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# SECTION 4: Parallel processing of video parts with SAM3 + merging
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 4 START: Parallel SAM3 pipeline processing"
echo "Start time: $START_TIME"
echo "====================================================="

PARTS=5
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BASE_DIR="outputs/${TIMESTAMP}_parts"
mkdir -p "$BASE_DIR"

echo "Will process video in $PARTS parts, base directory: $BASE_DIR"

for i in $(seq 1 $PARTS); do
    PART_DIR="$BASE_DIR/part_$(printf "%02d" $i)"
    mkdir -p "$PART_DIR"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Launching part $i/$PARTS -> $PART_DIR (background)"
    (
        source .venv/bin/activate
        python sam3_pipeline.py \
            --video test.mp4 \
            --prompt trash \
            --skip 10 \
            --part "$i/$PARTS" \
            --output "$PART_DIR"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Part $i finished."
    ) &
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for all parts to finish..."
wait
echo "$(date '+%Y-%m-%d %H:%M:%S') All parts completed."

MERGED_DIR="$BASE_DIR/merged"
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting merge of all parts into $MERGED_DIR"
python merger.py --dirs "$BASE_DIR"/part_* --output "$MERGED_DIR"
echo "$(date '+%Y-%m-%d %H:%M:%S') Merge completed."

LATEST_RUN="$MERGED_DIR"
echo "Latest run directory (merged): $LATEST_RUN"

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_4" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 4 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# SECTION 5: Classify frames
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 5 START: Classification"
echo "Start time: $START_TIME"
echo "====================================================="

python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_5" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 5 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# SECTION 6: YOLO training
# =============================================================================
SECTION_START=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "====================================================="
echo "SECTION 6 START: YOLO training"
echo "Start time: $START_TIME"
echo "====================================================="

cd "${LATEST_RUN}/yolo_training" || exit 1

yolo detect train \
  model="$YOLO_MODEL_PATH" \
  data=data.yaml \
  epochs=100 \
  imgsz=640

echo "Uploading trained model:"
python ../../../../upload_model.py --model ../../../../runs/detect/train/weights/best.pt

SECTION_END=$(date +%s)
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((SECTION_END - SECTION_START))
log_benchmark "SECTION_6" "$START_TIME" "$END_TIME" "$DURATION"   # BENCHMARK
echo "====================================================="
echo "SECTION 6 END: Duration: ${DURATION} seconds"
echo "====================================================="

# =============================================================================
# Overall script duration (optional extra benchmark line)
# =============================================================================
echo ""
echo "All sections completed successfully at $(date '+%Y-%m-%d %H:%M:%S')."
echo "Benchmark data written to $BENCHMARK_LOG"