#comparing the flops computing for phi-2

import transformers as tf
import fvcore as fv
from fvcore.nn import FlopCountAnalysis
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fvcore.nn import flop_count_table
from fvcore.nn import flop_count_str

torch.set_default_device("cpu")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
input_text = "Please give me a detailed information about the capital of France."

input_tokens = tokenizer(input_text, return_tensors="pt").input_ids
flops = FlopCountAnalysis(model, input_tokens)


#print(flops.total())
#print(flops.by_module())
#print(flops.by_operator())
#print(flops.by_module_and_operator())

print(flop_count_str(flops))
print(flop_count_table(flops))


