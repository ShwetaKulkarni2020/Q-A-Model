

# Q-A-Model

## Overview
This application was developed for IIT students to supply study materials. Focus is on physics materials only here. 
Physics text book is given as document to refer in the RAG pipeline. Since OpenAI is closed model and can't be fine tuned using Lora, we have used gpt2 model here. The query,key,value and o layers are choosen for lora config. The gradient flow Trainable parameters are set. And the fine tuned model is loaded with lora config.

## Tech Stack
- Python 
- PyTorch / Hugging Face Transformers
- Faiss vector database
- PEFT (LoRA/QLoRA)



## Problem Statement
IIT students run the app asking for relevant questions and they get the most accurate answer.


## Approach
- Data preprocessing
- Model / algorithm used is gpt2 and lora
- Training strategy ( supervised training )
- Reinforcement learning not done here...Could be used for better dialogue quality.


## How to Run
```bash
pip install torch transformers datasets peft PyPDF2 langchain faiss-cpu
python vector_db_faiss_physics_loraeffect.py
python prompty.py

