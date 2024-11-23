# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


cache_path='/disk2/yiran/hf_cache'
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl",cache_dir=cache_path)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl",cache_dir=cache_path)