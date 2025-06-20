FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["tritonserver", "--model-repository=/models"]
