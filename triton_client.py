import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import torch
from transformers import GPT2Tokenizer
import json
import os
import argparse

class TritonTextNormalizer:
    def __init__(self, url="localhost:8000", model_name="text_normalization"):
        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url)
        self.tokenizer = GPT2Tokenizer.from_pretrained("saarus72/russian_text_normalizer", eos_token='</s>')
        
    def preprocess_text(self, text):
        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]
        
    def postprocess_output(self, output_data):
        return self.tokenizer.decode(output_data[0], skip_special_tokens=True)
        
    def normalize_text(self, text):
        """
        Normalizes text using the model deployed on Triton Inference Server.
        
        Args:
            text (str): Input text for normalization
            
        Returns:
            str: Normalized text
        """
        try:
            input_ids, attention_mask = self.preprocess_text(text)
            
            inputs = []
            inputs.append(httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)))
            inputs[0].set_data_from_numpy(input_ids)
            
            inputs.append(httpclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)))
            inputs[1].set_data_from_numpy(attention_mask)
            
            results = self.client.infer(self.model_name, inputs)
            
            output_data = results.as_numpy("output")
            normalized_text = self.postprocess_output(output_data)
            
            return normalized_text
            
        except Exception as e:
            print(f"Error normalizing text with Triton: {e}")
            return text

def normalize_text(text, url="localhost:8000"):
    """
    Normalizes text using a model deployed on Triton Inference Server.
    
    Args:
        text (str): Input text for normalization
        url (str): Triton server URL
        
    Returns:
        str: Normalized text
    """
    try:
        client = httpclient.InferenceServerClient(url=url)
        
        tokenizer = GPT2Tokenizer.from_pretrained("ML_2/model_repository/text_normalization/1")
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].numpy()
        
        triton_input = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
        triton_input.set_data_from_numpy(input_ids)
        
        response = client.infer("text_normalization", [triton_input])
        
        output_ids = response.as_numpy("output")
        
        normalized_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return normalized_text
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test Triton Inference Server')
    parser.add_argument('--text', type=str, required=True, help='Text to normalize')
    parser.add_argument('--url', type=str, default='localhost:8000', help='Triton server URL')
    args = parser.parse_args()
    
    result = normalize_text(args.text, args.url)
    if result:
        print(f"Original text: {args.text}")
        print(f"Normalized text: {result}")

if __name__ == "__main__":
    main()
