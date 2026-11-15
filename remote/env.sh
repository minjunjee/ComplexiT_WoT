# cs103에서 쓰는 환경. pipeline.sh가 읽는다. GPU는 물리 번호 0, 1만 쓴다.
export CWOT_ROOT=/data/daehan/Introduction_LLM/ComplexiT_WoT
export HF_HOME=/data/daehan/hf_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$CWOT_ROOT/code
export TMPDIR=/data/daehan/tmp
# pip, uv 캐시도 /data 에 둔다(홈이 있는 / 는 여유가 거의 없다).
export PIP_CACHE_DIR=/data/daehan/tmp/pip_cache
export UV_CACHE_DIR=/data/daehan/tmp/uv_cache
if [ -x "$CWOT_ROOT/venv/bin/python" ]; then PY=$CWOT_ROOT/venv/bin/python; else PY=python3; fi
