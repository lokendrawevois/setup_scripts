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

git clone https://github.com/lokendrawevois/pipeline.git

echo "SAM3 download section"

if [ ! -f pipeline/sam3.pt ]; then
    wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt
else
    echo "sam3.pt already exists, skipping download."
fi

echo "SAM3 download section end"

if [ ! -f pipeline/yolo11n.pt ]; then
    wget https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt -O pipeline/yolo11n.pt
else
    echo "yolo11n.pt already exists, skipping download."
fi

cd pipeline

uv venv --python 3.9.6

source .venv/bin/activate

uv pip install -r requirements.txt

python download_training_data.py

# ===== NEW: Split processing into parts and merge =====
PARTS=5   # Number of parts (can be changed)

# Create a timestamped base directory for this run (same style as before)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BASE_DIR="outputs/${TIMESTAMP}_parts"
mkdir -p "$BASE_DIR"

echo "Processing video in $PARTS parts, base dir: $BASE_DIR"

# Run sam3_pipeline.py for each part
for i in $(seq 1 $PARTS); do
    PART_DIR="$BASE_DIR/part_$(printf "%02d" $i)"
    mkdir -p "$PART_DIR"
    echo "Running part $i/$PARTS -> $PART_DIR"
    python sam3_pipeline.py \
        --video test.mp4 \
        --prompt trash \
        --skip 10 \
        --part "$i/$PARTS" \
        --output "$PART_DIR"
done

# Merge all parts
MERGED_DIR="$BASE_DIR/merged"
echo "Merging all parts into $MERGED_DIR"
python merger.py --dirs "$BASE_DIR"/part_* --output "$MERGED_DIR"

# Set LATEST_RUN to the merged directory (for subsequent steps)
LATEST_RUN="$MERGED_DIR"
echo "Latest run directory (merged): $LATEST_RUN"
# ===== End of new section =====

# Continue with classification and training as before
python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

cd "${LATEST_RUN}/yolo_training" || exit 1

yolo detect train \
  model="$YOLO_MODEL_PATH" \
  data=data.yaml \
  epochs=100 \
  imgsz=640

echo "Uploading trained model:"
python ../../../upload_model.py --model ../../../runs/detect/train/weights/best.pt