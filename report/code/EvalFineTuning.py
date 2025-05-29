import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from FineTuning import FineTuningDataset, logger
import pandas as pd
import logging
from datetime import datetime
import random
import re

# Setup logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

#logger, log_file = setup_logging()

class Eval:
    def __init__(self, model_path="outputs", model_name="cjvt/GaMS-1B"):
        logger.info("📥 Loading dataset...")
        df = pd.read_csv("report/code/trainingdataset2.csv")
        finetuning_dataset = FineTuningDataset(df)
        self.dataset = finetuning_dataset.generate_dataset()
        
        logger.info("🤖 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_test_text(self, prompt, max_length=200):
        result = self.pipeline(prompt, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
        return result[0]['generated_text']

    def extract_prompt_and_response(self, text):
        user_match = re.search(r'<\|user\|>\s*(.*?)\s*<\|assistant\|>', text, re.DOTALL)
        assistant_match = re.search(r'<\|assistant\|>\s*(.*)', text, re.DOTALL)
        
        user_msg = user_match.group(1).strip() if user_match else ""
        assistant_msg = assistant_match.group(1).strip() if assistant_match else "N/A"
        
        return user_msg, assistant_msg


    def evaluate_x_examples(self, x=5):
        if not self.dataset:
            logger.warning("⚠️ No dataset provided for evaluation.")
            return

        logger.info(f"🔍 Evaluating {x} random examples from dataset...")

        # Convert to list of dicts if it's a HuggingFace Dataset object
        samples = self.dataset if isinstance(self.dataset, list) else self.dataset.to_list()
        selected_samples = random.sample(samples, min(x, len(samples)))

        for idx, sample in enumerate(selected_samples):
            print("SAMPLE: ", sample)
            text = sample.get('text', '')
            user_prompt, assistant_response = self.extract_prompt_and_response(text)

            logger.info(f"\n🧪 Example {idx + 1}" + "-"*50)
            logger.info(f"📝 Prompt: {user_prompt}")
            logger.info(f"🎯 Expected: {assistant_response}")

            try:
                generated = self.generate_test_text(prompt=user_prompt, max_length=500)
                logger.info(f"🤖 Generated: {generated}")

            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")

if __name__ == "__main__": 
    e = Eval()
    e.evaluate_x_examples(5)
