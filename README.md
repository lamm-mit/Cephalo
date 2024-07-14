# Cephalo: Multi-Modal Vision-Language Models for Bio-Inspired Materials Analysis and Design

Cephalo is a series of multimodal materials science focused vision large language models (V-LLMs) designed to integrate visual and linguistic data for advanced understanding and interaction in human-AI or multi-agent AI frameworks. 

A novel aspect of Cephalo's development is the innovative dataset generation method. The extraction process employs advanced algorithms to accurately detect and separate images and their corresponding textual descriptions from complex PDF documents. It involves extracting images and captions from PDFs to create well-reasoned image-text pairs, utilizing large language models (LLMs) for natural language processing. These image-text pairs are then refined and validated through LLM-based NLP processing, ensuring high-quality and contextually relevant data for training. 

Cephalo can interpret complex visual scenes and generating contextually accurate language descriptions and answer queries. 

The models are developed to process diverse inputs, including images and text, facilitating a broad range of applications such as image captioning, visual question answering, and multimodal content generation. The architecture combines a vision encoder model and an autoregressive transformer to process complex natural language understanding. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/kl5GWBP9WS0D4uwd1t3S7.png)

Cephalo provides a robust framework for multimodal interaction and understanding, including the development of complex generative pipelines to create 2D and 3D renderings of material microstructures as input for additive manufacturing methods.

## Information about this repository

Models are provided at [https://huggingface.co/lamm-mit/cephalo/](https://huggingface.co/lamm-mit/cephalo/). This repository provides additional codes, tools and analysis associated with the models. 

## Getting Started: Inference

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lamm-mit/Cephalo/blob/main/Cephalo%20Inference%20Colab.ipynb)

Open the Colab notebook above, or follow the instructions at [https://huggingface.co/lamm-mit/cephalo](https://huggingface.co/lamm-mit/cephalo) to get the model running on your local machine.

A simple example:


```python
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

question = "What is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI."

messages = [ 
    {"role": "user", "content": f"<|image_1|>\n{question}"}, 
    ] 

url = "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg" 

image = Image.open(requests.get(url, stream=True).raw) 

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

generation_args = { 
                    "max_new_tokens": 512, 
                    "temperature": 0.1, 
                    "do_sample": True, 
                    "stop_strings": ['<|end|>',
                                     '<|endoftext|>'],
                    "tokenizer": processor.tokenizer,
                  } 

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response) 
```
## Model merging and Training Script

- Model merging and training script: [Cephalo Model Merging 10b.ipynb](https://github.com/lamm-mit/Cephalo/blob/main/Cephalo%20Model%20Merging%20-%2010b.ipynb)

## Overview of Models

###  4b models

- [Cephalo-Phi-3-vision-128k-4b-alpha](https://huggingface.co/lamm-mit/Cephalo-Phi-3-vision-128k-4b-alpha)
  - Base version of the Cephalo-Phi-3 model, trained on GPT-4o distilled image-text data from Wikipedia and scientific papers. Good baseline model, but struggles in longer conversations. Context length of 128,000 tokens. 
- [Cephalo-Phi-3-vision-128k-4b-beta](https://huggingface.co/lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta)
  - Improved version of the Cephalo-Phi-3 model, trained on GPT-4o and Idefics-2 distilled image-text data from Wikipedia and scientific papers, as well as a large text-only corpus. Provides nuanced responses, with excellent reasoning. Context length of 128,000 tokens. 

### 8b models

- [Cephalo-Idefics-2-vision-8b-alpha](https://huggingface.co/lamm-mit/Cephalo-Idefics-2-vision-8b-alpha)
  - Trained on Idefics-2 distilled image-text data from Wikipedia and scientific papers. Gives shorter answers, to the point, and generaly accurate.
- [Cephalo-Idefics-2-vision-8b-beta](https://huggingface.co/lamm-mit/Cephalo-Idefics-2-vision-8b-beta)
  - Trained on GPT-4o distilled image-text data from Wikipedia and scientific papers. Gives longer answers, with enhanced reasoning. Can struggle with complex concepts.  
- [Cephalo-Llava-v1.6-Mistral-8b-alpha](https://huggingface.co/lamm-mit/Cephalo-Llava-v1.6-Mistral-8b-alpha)
  - Trained on GPT-4o distilled image-text data from Wikipedia, with low-resolution images. Does not perform well on multiple image queries, and has some inconsistencies in understanding.  

### Merged 10b models

- [Cephalo-Idefics-2-vision-10b-alpha](https://huggingface.co/lamm-mit/Cephalo-Idefics-2-vision-10b-alpha)
  - Merged model, 32+8=40 layers, checkpoint after first epoch. Trained on GPT-4o distilled image-text data from Wikipedia and scientific papers.
- [Cephalo-Idefics-2-vision-10b-beta](https://huggingface.co/lamm-mit/Cephalo-Idefics-2-vision-10b-beta)
  - Merged model, 32+8=40 layers, checkpoint after second epoch. Trained on GPT-4o distilled image-text data from Wikipedia and scientific papers.

### Merged 12b models

- [lamm-mit/Cephalo-Idefics-2-vision-12b-alpha](https://huggingface.co/lamm-mit/Cephalo-Idefics-2-vision-12b-alpha)
  - Merged model, 32+16=48 layers, checkpoint after first epoch. Trained on GPT-4o distilled image-text data from Wikipedia and scientific papers (dataset derivived from both Idefics-2 and GPT-4o distillation of the paper corpus).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/3Nfhn3f3FyK7Zgdg9GKJQ.png)

The image shows a summary of model merging approach, constructing larger models from smaller pre-trained building blocks. a, Fine-tuning the base model. b, Constructing the larger, merged model by combining the whole or parts of smaller models. c, Fine-tuning the integrated hybrid, merged, model.  

### Mixture-of-Experts models (including training and cookbook)

- [lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta](https://huggingface.co/lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta)
  - Mixture-of-expert model based on several smaller Cephalo-Phi-3 models.

A cookbook on how to create a MoE model from scratch is provided at: [https://huggingface.co/blog/mjbuehler/phi-3-vision-cephalo-moe](https://huggingface.co/blog/mjbuehler/phi-3-vision-cephalo-moe).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/623ce1c6b66fedf374859fe7/NK9KNOxmnVtn_PzwJtKPR.png)

### Training script 

A training script is provided here: [Cephalo_Fine-Tune.ipynb](https://github.com/lamm-mit/Cephalo/blob/main/Cephalo_Fine-Tune.ipynb). 

![image](https://github.com/user-attachments/assets/7338974e-b374-4ace-8e28-cfb594a309be)

### Fast inference with mistral.rs

Start an OpenAI comptabile server in a terminal:
```
./mistralrs_server --port 1234 vision-plain -m lamm-mit/Cephalo-Phi-3-vision-128k-4b-beta -a phi3v
```
Then, you can use it as follows:
```
import openai
import httpx
import textwrap, json

def log_response(response: httpx.Response):
    request = response.request
    print(f"Request: {request.method} {request.url}")
    print("  Headers:")
    for key, value in request.headers.items():
        if key.lower() == "authorization":
            value = "[...]"
        if key.lower() == "cookie":
            value = value.split("=")[0] + "=..."
        print(f"    {key}: {value}")
    print("  Body:")
    try:
        request_body = json.loads(request.content)
        print(textwrap.indent(json.dumps(request_body, indent=2), "    "))
    except json.JSONDecodeError:
        print(textwrap.indent(request.content.decode(), "    "))
    print(f"Response: status_code={response.status_code}")
    print("  Headers:")
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            value = value.split("=")[0] + "=..."
        print(f"    {key}: {value}")


openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

# Enable this to log requests and responses
# openai.http_client = httpx.Client(
#     event_hooks={"request": [print], "response": [log_response]}
# )

completion = openai.chat.completions.create(
    model="phi3v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "<|image_1|>\nWhat is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI.",
                },
            ],
        },
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0.3,
)
resp = completion.choices[0].message.content
print(resp)
```
More details on mistral.rs (including more sample scripts), see: https://github.com/EricLBuehler/mistral.rs

### Additional codes and tools

Additional codes and tools are provided at [https://github.com/lamm-mit/Cephalo](https://github.com/lamm-mit/Cephalo).
 
## Citation

Please cite as:

```bibtex
@article{Buehler_Cephalo_2024,
    title   = {Cephalo: Multi-Modal Vision-Language Models for Bio-Inspired Materials Analysis and Design},
    author  = {M.J. Buehler},
    journal = {arXiv},
    year    = {2024},
    volume  = {},
    pages   = {},
    url     = {https://arxiv.org/abs/2405.19076}
}
```
