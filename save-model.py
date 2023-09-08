import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def save_model_to_windows_folder():
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    # Create a Windows folder path for saving the model and tokenizer
    windows_folder_path = '/mnt/c/Users/mejan/projects/huggingface-models'  # Replace with your desired folder path

    # Create the folder if it doesn't exist
    if not os.path.exists(windows_folder_path):
        os.makedirs(windows_folder_path)

    # Save the model and tokenizer
    model.save_pretrained(windows_folder_path)
    tokenizer.save_pretrained(windows_folder_path)

    print(f"Model and tokenizer have been saved to {windows_folder_path}")

if __name__ == "__main__":
    save_model_to_windows_folder()
