python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1

req:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install openai-whisper
pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'

run:

1. python ./stt.py
2. python ./audiosplitter.py
