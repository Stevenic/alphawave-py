from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from alphawave_pyexts import serverUtils as sv
from auto_gptq import exllama_set_max_input_length

model_path = "/home/bruce/Downloads/models/Spicyboros-70b-22-GPTQ"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
model = exllama_set_max_input_length(model, 4096)
tokenizer = AutoTokenizer.from_pretrained(model_path)

#pipeline = transformers.pipeline(
#    "text-generation",
#    model=model,
##    tokenizer=tokenizer,
#    trust_remote_code=True,
#    device_map="auto",
#)
print(f"server started {model_path} into memory, use alpaca")

if __name__ == '__main__':
    #sv.server(model=None, tokenizer=tokenizer, pipeline = pipeline, stop_str=['User:'])
    sv.server(model=model, tokenizer=tokenizer, pipeline = None, stop_str=['User:', '###'])
