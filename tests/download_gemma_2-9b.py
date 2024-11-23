# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


cache_path='/disk2/yiran/hf_cache'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it",cache_dir=cache_path)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it",cache_dir=cache_path)