import torch
import transformers
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from alphawave_pyexts import serverUtils as sv


from huggingface_hub import snapshot_download

model_name = "tiiuae/falcon-40b-instruct"
#model_name = 'ehartford/WizardLM-Uncensored-Falcon-40b'
snapshot_download(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hello my name is"
device = "cuda:0"

inputs = tokenizer(text, return_token_type_ids=False, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)

sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
#sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:'])
