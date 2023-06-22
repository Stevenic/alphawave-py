from transformers import AutoModelForCausalLM, AutoTokenizer
from alphawave_pyexts import serverUtils as sv
from threading import Event, Thread

if __name__ == '__main__':
    model_name = "/user/bruce/FastChat/Falcon-40B-Instruct"
    #model_name='falcon-7b-instruct'
    stop_event = Event()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    #tokenizer.add_special_tokens(allowed=('<|endoftext|>'))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.tie_weights()
    print(f"Successfully loaded the model {model_name} into memory")
    streamer = MyStreamer(tokenizer, stop_event=stop_event)
    print('**** ready to serve on port 5004')

    sv.server_program()
