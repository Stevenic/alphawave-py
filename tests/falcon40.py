from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import accelerate

quantized_model_dir = "/home/bruce/Downloads/FastChat/falcon-40b-instruct-GPTQ/"
model_basename = "gptq_model-4bit--1g"

use_triton = True
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                                           use_safetensors=True,
                                           model_basename=model_basename,
                                           device="auto",
                                           #load_in_4bit=True,
                                           use_triton=use_triton,
                                           quantize_config=None,
                                           trust_remote_code=True)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

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

#sv.server(model=model, tokenizer=tokenizer, stop_str=['User:'])
