#pip3 install -U bitsandbytes scikit-learn
#pip3 install -U git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git

import os
import bitsandbytes as bnb
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

openai.api.key = ""
all_outputs = []

os.environ["CUDA_VISIBLE_DEVICES"]="0"
peft_model_id = "Amirkid/spotify-llama65B-Qlora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


def get_outputs(list1):
  
  for item in list:
    batch = tokenizer(item, return_tensors='pt')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=50)

    final_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    all_outputs.append(final_output)
    
    
def generate_questions_list():
  content = "Generate random questions which can be used to evaluate the performance of a Large Language model. Please number them"
  response = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages = [{"role": "user", "content": content}])
  
  message = response['choices'][0]['message']['content']
  
  return message


    
