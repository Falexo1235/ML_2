FROM python:3.11-slim

WORKDIR /app

COPY dist/*.whl .

RUN pip install --no-cache-dir *.whl

COPY ML_2/*.py ./ML_2/
COPY ML_2/wandb_config.yml ./ML_2/

ENV PYTHONPATH=/app/ML_2
ENV PYTHONUNBUFFERED=1

WORKDIR /app/ML_2

ENTRYPOINT ["python", "model_imply.py"]