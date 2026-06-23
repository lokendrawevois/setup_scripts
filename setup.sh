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

python sam3_pipeline.py --video train.mp4 --prompt trash --skip 10

LATEST_RUN=$(ls -td outputs/*/ | head -n 1)

echo "Latest run directory: $LATEST_RUN"

python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

cd "${LATEST_RUN}yolo_training" || exit 1

yolo detect train \
  model="$YOLO_MODEL_PATH" \
  data=data.yaml \
  epochs=100 \
  imgsz=640

echo "Uploading trained model:"
python ../../../upload_model.py --model ../../../runs/detect/train/weights/best.pt
