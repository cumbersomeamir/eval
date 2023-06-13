#pip3 install -U bitsandbytes scikit-learn
#pip3 install -U git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git git+https://github.com/huggingface/peft.git

import os
import bitsandbytes as bnb
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

openai.api.key = "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"
all_outputs = []

os.environ["CUDA_VISIBLE_DEVICES"]="0"
peft_model_id = "Amirkid/spotify-llama65B-Qlora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

#Generate Outputs Function
def get_outputs(list1):
  
  for item in list:
    batch = tokenizer(item, return_tensors='pt')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=50)

    final_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    all_outputs.append(final_output)
    return all_outputs
    
#Generate Questions Function    
def generate_questions_list(num_questions):
  openai.api_key = "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"
  content = f"Can you generate {num_questions} questions which can help the performance of LLMs?"
  response = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages = [{"role":"user", "content": content}])


  message = response['choices'][0]["message"]["content"]

  data = str(message)
  print(data)
  import re
  # Assuming 'data' is your string containing all the questions
  questions = re.split(r'\d+\.', data)
  questions = [question.strip() for question in questions if question.strip()]
  all_questions = all_questions + questions
  questions = []

  
  
  def main():
    all_questions = generate_questions_list(100)
    all_outputs = get_outputs(all_questions)
    final_dict = {"input" : all_questions , "output" : all_outputs)
    final_df = pd.DataFrame(final_dict)
 
main()                  
    

    
