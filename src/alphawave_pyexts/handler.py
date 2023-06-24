import torch

from typing import Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path=""):
        # load model and tokenizer from path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # preprocess
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.model.generate(**inputs, **parameters)
        else:
            outputs = self.model.generate(**inputs)

        # postprocess the prediction
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return [{"generated_text": prediction}]