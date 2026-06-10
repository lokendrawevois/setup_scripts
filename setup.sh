git clone https://github.com/lokendrawevois/pipeline.git

UV_VENV_CLEAR=1

wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt

wget https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt -O pipeline/yolo11n.pt

cd pipeline

uv venv --python 3.9.6

source .venv/bin/activate

uv pip install -r requirements.txt

.venv/bin/python full_pipeline.py --video sample.mp4 --prompt trash --skip 100

LATEST_RUN=$(ls -td outputs/*/ | head -n 1)

echo "Latest run directory: $LATEST_RUN"

.venv/bin/python classify.py \
    --run_dir "$LATEST_RUN" \
    --max_workers 10 \
    --frame_workers 10

cd "${LATEST_RUN}yolo_training" || exit 1

yolo detect train \
  model=../../../yolo11n.pt \
  data=data.yaml \
  epochs=10 \
  imgsz=640

echo "=== Training output ===, locally it won't point to this, just for runpod"
ls -lah "pipeline/runs/detect/train"
