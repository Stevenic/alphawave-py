from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from alphawave_pyexts import serverUtils as sv
from auto_gptq import exllama_set_max_input_length

model_path = "/home/bruce/Downloads/models/Spicyboros-70b-22-GPTQ"
aiboros_map={
    "model.embed_tokens.weight": 0,
    "model.layers.0": 0,"model.layers.1": 0,"model.layers.2": 0,"model.layers.3": 0,"model.layers.4": 0,"model.layers.5": 0, "model.layers.6": 0,
    "model.layers.7": 0,"model.layers.8": 0,"model.layers.9": 0,"model.layers.10": 0,"model.layers.11": 0,"model.layers.12": 0,"model.layers.13": 0,
    "model.layers.14": 0,"model.layers.15": 0,"model.layers.16": 0,"model.layers.17": 0,"model.layers.18": 0,"model.layers.19": 0,"model.layers.20": 0,
    "model.layers.21": 0,"model.layers.22": 0,"model.layers.23": 0,"model.layers.24": 0,"model.layers.25": 0,"model.layers.26": 0,"model.layers.27": 0,
    "model.layers.28": 0,"model.layers.29": 0,"model.layers.30": 0,"model.layers.31": 0,"model.layers.32": 0,"model.layers.33": 0,"model.layers.34": 0,
    "model.layers.35": 0,"model.layers.36": 0,"model.layers.37": 0,"model.layers.38": 0,"model.layers.39": 0,"model.layers.40": 1,"model.layers.41": 1,
    "model.layers.42": 1,"model.layers.43": 1,"model.layers.44": 1,"model.layers.45": 1,"model.layers.46": 1,"model.layers.47": 1,"model.layers.48": 1,
    "model.layers.49": 1,"model.layers.50": 1,"model.layers.51": 1,"model.layers.52": 1,"model.layers.53": 1,"model.layers.54": 1,"model.layers.55": 1,
    "model.layers.56": 1,"model.layers.57": 1,"model.layers.58": 1,"model.layers.59": 1,"model.layers.60": 1,"model.layers.61": 1,"model.layers.62": 1,
    "model.layers.63": 1,"model.layers.64": 1,"model.layers.65": 1,"model.layers.66": 1,"model.layers.67": 1,"model.layers.68": 1,"model.layers.69": 1,
    "model.layers.70": 1,"model.layers.71": 1,"model.layers.72": 1,"model.layers.73": 1,"model.layers.74": 1,"model.layers.75": 1,"model.layers.76": 1,
    "model.layers.77": 1,"model.layers.78": 1,"model.layers.79": 1,"model.norm.weight": 1,"lm_head.weight": 1
}

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=aiboros_map)
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
