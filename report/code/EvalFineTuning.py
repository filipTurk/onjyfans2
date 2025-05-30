import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from FineTuning import FineTuningDataset, logger
import pandas as pd
import logging
from datetime import datetime
import random
import re

#change datasetfilepath beforerun
#changemodelname

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
    def __init__(self, model_path="outputs", model_name="cjvt/GaMS-9B-Instruct"):
        logger.info("📥 Loading dataset...")
        df = pd.read_csv("ul-fri-nlp-course-project-2024-2025-onjyfans/report/code/trainingdataset2.csv")
        self.original_df = df  # Keep original for CSV export
        finetuning_dataset = FineTuningDataset(df)
        self.dataset = finetuning_dataset.generate_dataset()
        
        logger.info("🤖 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_test_text(self, prompt, max_length=200):
        # Stop generation before going into the assistant's message again
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        generated_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # More deterministic for evaluation
            pad_token_id=self.tokenizer.eos_token_id
        )
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return result


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

        samples = self.dataset if isinstance(self.dataset, list) else self.dataset.to_list()
        selected_samples = random.sample(samples, min(x, len(samples)))

        for idx, sample in enumerate(selected_samples):
            text = sample.get('text', '')
            user_prompt, assistant_response = self.extract_prompt_and_response(text)

            logger.info(f"\n🧪 Example {idx + 1}" + "-"*50)
            logger.info(f"📝 Prompt: {user_prompt}")
            logger.info(f"🎯 Expected: {assistant_response}")

            try:
                prompt = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
                generated = self.generate_test_text(prompt=prompt, max_length=500)
                generated = generated.split("<|assistant|>")[-1].strip()
                logger.info(f"🤖 Generated: {generated}")
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")

    def generate_evaluation_csv(self, output_path="evaluation.csv", num_examples=20):
        logger.info(f"📄 Generating evaluation CSV with {num_examples} examples...")

        samples = self.dataset if isinstance(self.dataset, list) else self.dataset.to_list()
        selected_samples = random.sample(samples, min(num_examples, len(samples)))
        eval_rows = []

        for sample in selected_samples:
            text = sample.get('text', '')
            user_prompt, expected_response = self.extract_prompt_and_response(text)

            try:                                                                                                                                                                                       
                prompt = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
                generated_response = self.generate_test_text(prompt=prompt, max_length=500) 
                generated_response = generated_response.split("<|assistant|>")[-1].strip()                                                         
            except Exception as e:
                logger.error(f"❌ Generation failed for one sample: {e}")
                generated_response = "Generation failed"

            eval_rows.append({
                "user_message": user_prompt,
                "expected_message": expected_response,
                "generated_message": generated_response
            })

        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(output_path, index=False)
        logger.info(f"✅ Evaluation CSV saved to: {output_path}")

if __name__ == "__main__": 
    e = Eval()
    e.evaluate_x_examples(5)  # Optional: log samples
    e.generate_evaluation_csv(output_path="evaluation_gams9b.csv", num_examples=20)

