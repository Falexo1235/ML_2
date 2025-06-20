# Text Normalization for TTS (Russian)

## Author
Anisimov Aleksey, 972303

## Description
This project is a solution for the "Text Normalization Challenge - Russian Language" problem on Kaggle. The goal is to normalize russian text given in the csv file. The result is supposed to be used for tts.

## Project Structure
```
| ML_2/
|
|--| ML_2/
|  |-- mode_imply.py
|  |--| model_repository/
|  |  |--| text_normalization/
|  |     |-- config.pbtxt
|  |     |--| 0/
|  |--| data/
|     |-- log_file.log

|
|--| dist/
|  |-- machinelearning-0.1.0-py3-none-any.whl
|
|-- docker-compose.yml
|-- Dockerfile
|-- pyproject.toml
|-- poetry.lock
|-- README.md
|-- triton.Dockerfile
|-- triton_client.py
|
|--| notebooks/
|  |--rules_normalization.ipynb
|  |--improved_baseline.ipynb
|  |--neural_attempt.ipynb
```

## How to Use

### 1. Installing Dependencies

1. Ensure you have Python 3.11 or higher installed.
2. Install .whl file:
```bash
pip install machinelearning-0.1.0-py3-none-any.whl
```

### 2. Training and Normalization (Python)

**Train the model:**
```bash
python ML_2/model_imply.py --train
```

**Normalize text (default test set):**
```bash
python ML_2/model_imply.py --normalize
```

**Normalize with custom dataset:**
```bash
python ML_2/model_imply.py --normalize --input=ML_2/data/ru_test_2.csv --output=ML_2/data/final_submission.csv
```

### 3. Using Docker

**Build the Docker image:**
```bash
docker-compose build text-normalization
```

**Train the model in Docker:**
```bash
docker-compose run text_normalization --train
```

**Normalize with default data in Docker:**
```bash
docker-compose run text_normalization --normalize
```

**Normalize with custom data in Docker:**
```bash
docker-compose run -v /path/to/data:/app/ML_2/data text_normalization --normalize --input=/app/ML_2/data/ru_test_2.csv --output=/app/ML_2/data/final_submission.csv
```

### 4. Triton Inference Server

**Build and launch Triton server:**
```bash
docker-compose up triton
```

- The model will be available at:
  - HTTP: `localhost:8000`
  - gRPC: `localhost:8001`

**Client Example:**
See `triton_client.py` for a Python example of sending requests to the Triton server.

---

## My result progress:
| My model | Baseline | Improved baseline | With neural model |
| -------- | -------- | ----------------- | ----------------- |
| 0.88350  | 0.96465  |      0.97197      |      0.97589      |

## Resources used:
1. Rule baseline: https://www.kaggle.com/code/arccosmos/ru-baseline-lb-0-9799-from-en-thread
2. Additional data for baseline: https://www.kaggle.com/datasets/richardwilliamsproat/text-normalization-for-english-russian-and-polish
3. Baseline improvement: https://www.kaggle.com/code/jtoffler/baseline-improvement-lb-9750-to-9751
4. Neural text normalization: https://huggingface.co/saarus72/russian_text_normalizer
