from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
import torch
from accelerate import load_checkpoint_and_dispatch
import datetime
import os
import traceback
from alphawave_pyexts import serverUtils as sv

# Load the model.

quantized_model_dir = "/home/bruce/Downloads/FastChat/guanaco-33B-GPTQ"
model_basename = "guanaco-33B-GPTQ-4bit.act-order"
model_basename = "guanaco33-4bit-128g"

use_triton = True
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                                           #use_safetensors=True,
                                           model_basename=model_basename,
                                           device="auto",
                                           #load_in_4bit=True,
                                           use_triton=use_triton,
                                           quantize_config=None,
                                           trust_remote_code=True)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)
input('?')
prompt = "Tell me about AI"
prompt_template=f'''### Human: {prompt}
### Assistant:'''

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)


print(pipe(prompt_template)[0]['generated_text'])

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

"""
print('devices', torch.cuda.device_count(), torch.cuda.current_device())
print(f"Starting to load the model {model_name} into memory")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
model.tie_weights()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
"""
print(f"Successfully loaded the model {model_name} into memory")

if __name__ == '__main__':
    sv.server(pipeline=pipeline)
