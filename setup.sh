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

echo "$(date '+%Y-%m-%d %H:%M:%S') GEMINI_API_KEY detected."

git clone https://github.com/lokendrawevois/pipeline.git

wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt

wget https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt -O pipeline/yolo11n.pt

cd pipeline

uv venv --python 3.9.6

source .venv/bin/activate

uv pip install -r requirements.txt

python download_training_data.py

python full_pipeline.py --video sample.mp4 --prompt trash --skip 10

LATEST_RUN=$(ls -td outputs/*/ | head -n 1)

echo "Latest run directory: $LATEST_RUN"

python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

cd "${LATEST_RUN}yolo_training" || exit 1

yolo detect train \
  model=../../../yolo11n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640

BEST_MODEL=$(find runs/detect -path "*/weights/best.pt" -type f -exec ls -t {} + | head -n 1)

if [ -z "${BEST_MODEL:-}" ]; then
    echo "ERROR: Could not find YOLO best.pt after training."
    exit 1
fi

echo "Uploading trained model: $BEST_MODEL"

python ../../../upload_model.py \
  --model "$BEST_MODEL"
