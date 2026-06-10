if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "Please export it before running the script:"
    echo "export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') GEMINI_API_KEY detected."

git clone https://github.com/lokendrawevois/pipeline.git

wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt

cd pipeline

uv venv --python 3.9.6

source .venv/bin/activate

uv pip install -r requirements.txt

python full_pipeline.py --video sample.mp4 --prompt trash --skip 10

LATEST_RUN=$(ls -td outputs/*/ | head -n 1)

echo "Latest run directory: $LATEST_RUN"

python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

cd "${LATEST_RUN}yolo_training" || exit 1

yolo detect train \
  model=yolo11n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640
