import tritonclient.http as httpclient
import numpy as np
import argparse

def normalize_text(text, url="localhost:8000"):
    client = httpclient.InferenceServerClient(url=url)
    input_tensor = httpclient.InferInput("INPUT", [1], "BYTES")
    input_tensor.set_data_from_numpy(np.array([text.encode("utf-8")], dtype="|S"))
    response = client.infer("text_normalization", [input_tensor])
    # Получаем результат из outputs
    output = response.as_numpy("OUTPUT")[0].decode("utf-8")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Text Normalization Client")
    parser.add_argument("--input", type=str, required=True, help="Input text to normalize")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    result = normalize_text(args.input, url=args.url)
    print(result)