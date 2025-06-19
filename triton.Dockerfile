FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

RUN pip install --no-cache-dir transformers

CMD ["tritonserver", "--model-repository=/models"]
