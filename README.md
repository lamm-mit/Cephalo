# Cephalo

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

- Model merging and training  script: [Cephalo%20Model%20Merging%20-%2010b.ipynb](https://github.com/lamm-mit/Cephalo/blob/main/Cephalo%20Model%20Merging%20-%2010b.ipynb)

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

### Additional codes and tools

Additional codes and tools are provided at [https://github.com/lamm-mit/Cephalo](https://github.com/lamm-mit/Cephalo).
 
## Citation

Please cite as:

```bibtex
@article{Buehler_Cephalo_2024,
    title   = {Cephalo: Multi-Modal Vision-Language Models for Bio-Inspired Materials Analysis and Design},
    author  = {M.J. Buehler},
    journal = {},
    year    = {2024},
    volume  = {},
    pages   = {},
    url     = {}
}
```
