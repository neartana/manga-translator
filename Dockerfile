# Manga Translator — all-in-one image.
# Default build = lightweight CPU image (OpenCV fallbacks, no PyTorch).
# For the full ML stack:  docker build --build-arg INSTALL_ML=1 -t manga-translator .
# For GPU: build the ML image and run with --gpus all (needs nvidia-container-toolkit),
# then pick "GPU (CUDA)" in the UI or pass --device cuda to the CLI.

FROM python:3.11-slim

ARG INSTALL_ML=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY manga_translator ./manga_translator
COPY server ./server
COPY fonts ./fonts

EXPOSE 7860
VOLUME ["/app/outputs", "/app/models"]

CMD ["python", "-m", "manga_translator", "--serve", "--host", "0.0.0.0", "--port", "7860"]
