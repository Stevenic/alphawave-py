from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from alphawave_pyexts import serverUtils as sv
import inspect

"""
if __name__ == '__main__':
    model_name = "/home/bruce/Downloads/FastChat/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.tie_weights()
    print(f"Successfully loaded the model {model_name} into memory")
    print('**** ready to serve on port 5004')

    sv.server_program(model, tokenizer)
"""
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sv.server_program(model=None, tokenizer=None, pipeline = pipeline)
#sequences = pipeline(
#   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compar#ed to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
#    max_length=200,
#    do_sample=True,
#    top_k=10,
#    num_return_sequences=1,
#    eos_token_id=tokenizer.eos_token_id,
#)


