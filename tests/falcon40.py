import torch
import transformers
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from alphawave_pyexts import serverUtils as sv


from huggingface_hub import snapshot_download

#model_name = "tiiuae/falcon-40b-instruct"
model_name = 'ehartford/WizardLM-Uncensored-Falcon-40b'
print(f"Loading {model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hello my name is"
device = "cuda:0"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)
print(f"server started {model_name} into memory, use falcon_instruct, or wizardLM2 if using WizardLM...Falcon")

sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
#sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:'])
