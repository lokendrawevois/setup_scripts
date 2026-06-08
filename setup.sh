git clone https://github.com/lokendrawevois/pipeline.git

UV_VENV_CLEAR=1

wget https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt -O pipeline/sam3.pt

cd pipeline

uv venv --python 3.9.6

source .venv/bin/activate

uv pip install -r requirements.txt

.venv/bin/python full_pipeline.py --video sample.mp4 --prompt trash --skip 20
