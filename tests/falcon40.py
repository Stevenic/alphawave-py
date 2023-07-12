from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import accelerate

model_dir = "TheBloke/falcon-40b-instruct-GPTQ"
model_basename = "gptq_model-4bit--1g"

use_triton = True
tokenizer = AutoTokenizer.from_pretrained("TheBloke/falcon-40b-instruct-GPTQ")

model = AutoModelForCausalLM.from_pretrained("TheBloke/falcon-40b-instruct-GPTQ", quantize_config=None, use_safetensors=True, trust_remote_code=True)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)



sv.server(model=model, tokenizer=tokenizer, stop_str=['User:'])
