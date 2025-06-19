# Homework 2
Natural Language Processing and Triton Inference Servers.

__DEADLINE:  --.--.2025__


## Time to get familliar with Natural Language Processing and Triton Inference Server

#### Learn more about NLP:
- https://github.com/yandexdataschool/nlp_course (ru lectures with seminars)
- https://lena-voita.github.io/nlp_course.html
- https://ods.ai/tracks/nlp-course-spring-2025
- https://ods.ai/tracks/df24-nlp
- https://web.stanford.edu/~jurafsky/slp3/
- https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#1-define-our-use-case (https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb)
- https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2025.ipynb

#### Triton inference server:
- https://github.com/triton-inference-server/tutorials

Get ready: NVIDIA's documentation can sometimes be incomplete or contain inaccuracies. 
Don`t forget check the official GitHub issues for known problems and workarounds, or ask any question at course chat.

### Intro

Many speech and language applications, including text-to-speech (TTS) and automatic speech recognition (ASR), require text normalization - converting written expressions into appropriate spoken forms (e.g., converting "12:47" to "twelve forty-seven" or "$3.16" to "three dollars, sixteen cents").
One of the major challenges when developing TTS or ASR systems for a new language is creating and testing grammar rules for these conversions, which requires significant linguistic expertise and native speaker intuition.

Here is the original competition:
[kaggle competition](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/overview)



### Task
Implement and deploy a production-ready text normalization model for TTS.
The current homework consists of two parts:
*  Deep Learning Engineer part.
*  Software Engineer part.
  
Firstly, you need to train a competitive  system, and secondly: wrap your code into a production-ready artifact, which may be deployed on any Linux server in one command.

--------------
### Requirements
We don't accept homework if any of the following requirements are not satisfied:
- The code should be situated in a public github (or gitlab) repository. Two branches: `dev` for development and `master` for the latest working version.
- You should build your project using [poetry](https://python-poetry.org/docs/) to freeze dependendecies of python packages you use. And attach your .whl file
- Readable and Understandable `README.md` file:
    - Your fullname & group
    - "**How To**" for your repo: train model, evaluate, deploy. With docker.
    - Resources you utilized
- Your code must be fully covered with logging to `./data/log_file.log`. The file should be viewable and downloadable
- Proper `.gitignore` file. You do not want rubbish in your repo.
- The major software artifact is `model_impl.py`, containing the class `My_TextNormalization_Model` with following method: 
  - ```def normalize text(text: str) -> str```

- `Dockerfile`
- `docker-compose.yaml`
- Submission on kaggle competition with Token Accuracy (the total percent of correct tokens) > 0.96.
Read more about the metric [here](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/overview/evaluation)
  
### Dataset
Here the competition on Kaggle with Dataset: [kaggle competition](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/)

BUT we have a problem, the data quite dirty and have mistakes. Highly recommend for you find the way to clean data markup.
You have options here:
* Use rules or heuristics to clean simple cases
* Do all clean up using LLM. I suggest you use free API LLM or any Local Model (run them using [vllm](https://github.com/vllm-project/vllm)). 

Free API:
  * [Gemini, Gemma](https://ai.google.dev/gemini-api/docs/rate-limits)
  * https://console.mistral.ai/api-keys 
  * any other you will find

  Also check Structured Output feature (All modern LLM APIs support it):
  - [vllm](https://docs.vllm.ai/en/latest/features/structured_outputs.html)

Also recommend to collaborate with your colleagues and find the way to clean data and share it.


### Project Milestones
##### 1. Data Science in Jupyter
Feel free to stick to Jupyter or Colab environment. Here we expect you to build two classifier models. 
 1) Baseline rule classifier. No ML, just rules with heuristics.
 2) Neural Text Normalization Model. Check T5, BART, etc. Or experiment with LLAMA/GPT/Mistral models with 0-shot, few-shot without finetuning, or try finetune LoRa adapter.

For both tasks, please, refer to target metrics at the end of README.
Pay attention, here you will create submissions on Kaggle platform using created your own notebook with your model.

Catch up the baseline.ipynb on rules with Kaggle submission format creation [kaggler solution](https://www.kaggle.com/code/arccosmos/ru-baseline-lb-0-9799-from-en-thread)

**Note!**: your model train process can stay in clean jupyter notebook file, you should pack to app **only** the model inference.

At Dark time use this hints:
1. Remember you can use any utils you will find at the internet. Any model from [huggingface] (https://huggingface.co/models)
2. [check linguists view for this task](https://dialogue-conf.org/media/3906/cherepanovaod.pdf)
3. https://github.com/NVIDIA/NeMo-text-processing 

##### 2. Pack into git repo
At this point we expect to see fully working application in the `master` branch.  
  
##### 3. Pack to production ready solution
###### 3.1 Triton Inference Server

To ensure high-performance deployment of your text normalization model, you need to package it for use with NVIDIA Triton Inference Server. Follow these instructions:

1. **Model Preparation**:
   - Export your trained model to ONNX format or save it in a format supported by Triton (PyTorch, TensorFlow, ONNX)
   - If you want to use LLM or achieve high performance using Encoder-Decoder models, I recommend converting your model to TensorRTLLM, follow the git repository [TensorRTLLM](https://github.com/NVIDIA/TensorRT-LLM) and the example [Llama3 with TensorRTLLM](https://www.infracloud.io/blogs/running-llama-3-with-triton-tensorrt-llm/)
   - Make sure all pre- and post-processing of data is documented

2. **Model Structure for Triton**:
   - Create a `model_repository` directory with the following structure:
     ```
     model_repository/
     └── text_normalization/
         ├── config.pbtxt
         └── 1/
             └── model.onnx  # or other model format
     ```
   - The `config.pbtxt` file should contain the model configuration, including:
     - Model name
     - Platform (ONNX, PyTorch, TensorFlow)
     - Input and output tensors with their shapes and data types
     - Optimization parameters

3. **Creating a Docker Image with Triton**:
   - Add a `triton.Dockerfile` to your repository:
     ```dockerfile
     FROM nvcr.io/nvidia/tritonserver:22.12-py3
     
     WORKDIR /app
     COPY model_repository /models
     
     # If additional dependencies are required
     COPY requirements.txt .
     RUN pip install -r requirements.txt
     
     # Run Triton server
     CMD ["tritonserver", "--model-repository=/models"]
     ```

4. **Updating docker-compose.yaml**:
   - Add the Triton service to your docker-compose.yaml:
     ```yaml
     services:
       triton:
         build:
           context: .
           dockerfile: triton.Dockerfile
         ports:
           - "8000:8000"  # HTTP API
           - "8001:8001"  # gRPC API
           - "8002:8002"  # Metrics
         volumes:
           - ./model_repository:/models
         deploy:
           resources:
             reservations:
               devices:
                 - driver: nvidia
                   count: 1
                   capabilities: [gpu]
     ```

5. **Client Code**:
   - Create a `triton_client.py` file to interact with the server:
     ```python
     import tritonclient.http as httpclient
     from tritonclient.utils import np_to_triton_dtype
     import numpy as np
     
     def normalize_text(text, url="localhost:8000"):
         """
         Normalizes text using a model deployed on Triton Inference Server.
         
         Args:
             text (str): Input text for normalization
             url (str): Triton server URL
             
         Returns:
             str: Normalized text
         """
         client = httpclient.InferenceServerClient(url=url)
         
         # Text preprocessing (adapt to your model)
         input_data = preprocess_text(text)
         
         # Prepare input data
         inputs = []
         inputs.append(httpclient.InferInput("input_ids", input_data.shape, np_to_triton_dtype(input_data.dtype)))
         inputs[0].set_data_from_numpy(input_data)
         
         # Send request
         results = client.infer("text_normalization", inputs)
         
         # Get and post-process results
         output_data = results.as_numpy("output")
         normalized_text = postprocess_output(output_data)
         
         return normalized_text
     ```

6. **Testing and Optimization**:
   - Start the Triton server: `docker-compose up triton`
   - Test performance using Triton Performance Analyzer [Triton Performance Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md) + [nvidia docs about perf analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2280/user-guide/docs/user_guide/perf_analyzer.html)
   - Your app 
   - Optimize the model configuration to achieve the best performance:
     - Configure dynamic batching parameters
     - Experiment with parallelism parameters
     - Consider using TensorRTLLM for additional optimization

7. **Documentation**:
   - Update README.md with instructions for deployment and using Triton
   - Include examples of API requests and expected responses
   - Document performance metrics (latency, throughput)

Using Triton Inference Server will allow your model to process requests with high throughput and low latency, which is critical for industrial TTS systems.


##### 4. Grades
###### 4.1 Data Science part  
| Points         | Token Accuracy     | Description |
|--------------|-----------|------------|
| 0       | < 0.5      |        |
| 10      | [0.9; 0.96)| Good Baseline.       |
| 15      | [0.96; 0.97] | Close to SOTA      |
| 15      | > 0.97  |  SOTA?       |  You can create your own test set and show, that your model works great, but test set on kaggle is bad. And get 10 points for this section.


__Total: 40 points__  
Please, note that cheating with metrics will lead you to the grade 0.

###### 4.2 Software Engineer part  
  
| Points         | Bulletpoint     | Description |
|--------------|-----------|------------|
| 15  | Triton Inference Server | Model successfully wrapped in Triton Inference Server with proper configuration. Server starts and processes requests.  |
| 10 | Perf Analyzer | Performance measurements conducted and documented using Triton Perf Analyzer. Latency and throughput metrics are presented. |
| 10  | [ONNX](https://github.com/onnx/onnx) or TensorRTLLM model inference | You should use .onnx format for you model to speed up the inference of your model. You can use optimum to convert your model to onnnx format and run the inference. Hints: [1](https://www.philschmid.de/convert-transformers-to-onnx); [2](https://github.com/huggingface/optimum) |
|  5   |Code quality   | Clear OOP pattern. Well in-code comments. Build-in documentation for each function. No code duplicates. Meaningful variable names       |
|  5   | model.py      |    The model is properly packed into the class inside *.py file.      |
| 5 | Wandb Your Model Training Artifacts  | You log all your model train process using [Wandb](https://wandb.ai/site) or ClearML |
|    2   | Logging       |Catch and log all possible errors. Singleton logging pattern (use logging module)      |
|   3    | git workflow  | Publicly available repo. dev and master branches. Regular Commits. No Commit Rush. Meaningful comment for each commit.    |
|    5   |docker-compose | working docker-compose file       |


__Total: 60 points__ 



###### Bonus Part
You can find Data Science article in the field you want to go deeper (e.x Natural Language Processing, Computer Vision, Reinforcement Learning). If you want, you can always ask me about help and we will find together interesting article for you.

After you read whole article you can prepare fast review and we can discuss it together, like it happens in [DS Talks Siberia seminars](https://t.me/+fQ07VSVJ2V8yZGYy), seminars recordings available on [Youtube](https://www.youtube.com/channel/UCKi44xqXU67E3dv5e0b_0Dg).

__Total: 20 points__  Over all points at this task.
