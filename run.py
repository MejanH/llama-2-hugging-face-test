import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timeStart = time.time()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    low_cpu_mem_usage=True,
)

# Move the model to the device
model.to(device)

print("Load model time: ", -timeStart + time.time())

while(True):
    input_str = input('Enter: ')
    input_token_length = input('Enter length: ')

    if(input_str == 'exit'):
        break

    timeStart = time.time()

    inputs = tokenizer.encode(
        input_str,
        return_tensors="pt"
    )

    # Move the inputs to the device
    inputs = inputs.to(device)

    outputs = model.generate(
        inputs,
        max_length=int(input_token_length) + len(inputs[0]),
        pad_token_id=tokenizer.eos_token_id
    )

    output_str = tokenizer.decode(outputs[0])

    print(output_str)

    print("Time taken: ", -timeStart + time.time())
