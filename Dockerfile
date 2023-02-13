FROM  python:3.7.13-slim-buster

WORKDIR /app

COPY . .
RUN pip install pip==22.1.2
RUN pip install -r requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
CMD ["python", "./run_api.py"]
