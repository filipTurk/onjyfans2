import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import random
from datasets import Dataset
import json
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
import logging
import sys
import time
from datetime import datetime
import os

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# Initialize logging
logger, log_file = setup_logging()

class FineTuner:
    def __init__(self, modelName="cjvt/GaMS-2B", tokenizerName="cjvt/GaMS-2B", dataset=None, debug_mode=True):
        self.debug_mode = debug_mode
        self.model_name = modelName
        self.tokenizer_name = tokenizerName
        self.dataset = dataset
        
        logger.info(f"🚀 Starting FineTuner initialization")
        logger.info(f"Model: {modelName}")
        logger.info(f"Dataset size: {len(dataset) if dataset else 'None'}")
        logger.info(f"Debug mode: {debug_mode}")
        
        start_time = time.time()
        
        try:
            logger.info("📦 Loading BitsAndBytes config...")
            self.bnb_config = self.loadBnBConfig()
            
            logger.info("🤖 Loading model...")
            self.load_model(model_name=modelName)
            logger.info(f"✅ Model loaded in {time.time() - start_time:.1f}s")
            
            logger.info("🔤 Loading tokenizer...")
            self.load_tokenizer()
            logger.info("✅ Tokenizer loaded")
            
            # Test generation with base model
            logger.info("🧪 Testing base model generation...")
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            test_output = self.generate_test_text(prompt="Zdravo, kako si?")
            logger.info(f"Base model test: {test_output[:100]}...")
            
            # LoRA setup
            logger.info("🔧 Setting up LoRA configuration...")
            self.lora_alpha = 32
            self.lora_dropout = 0.1
            self.lora_r = 16
            
            logger.info("🎯 Applying LoRA to model...")
            self.lora_model = get_peft_model(self.model, self.loadPeftConfig())
            
            logger.info("📊 Model parameters:")
            self.print_trainable_parameters(self.model, "Base model")
            self.print_trainable_parameters(self.lora_model, "LoRA model")
            
            # Test LoRA model
            logger.info("🧪 Testing LoRA model generation...")
            self.pipeline = pipeline("text-generation", model=self.lora_model, tokenizer=self.tokenizer)
            test_output = self.generate_test_text(prompt="Hello, how are you?", max_length=50)
            logger.info(f"LoRA model test: {test_output[:100]}...")
            
            logger.info("🏋️ Initializing trainer...")
            self.trainer = self.init_trainer()
            logger.info("✅ Trainer initialized successfully")
            
            # Only train if not in debug mode
            if not self.debug_mode:
                logger.info("🎓 Starting training...")
                train_start = time.time()
                self.trainer.train()
                train_time = time.time() - train_start
                logger.info(f"✅ Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
                
                logger.info("📈 Evaluating model...")
                metrics = self.trainer.evaluate()
                logger.info(f"Evaluation metrics: {metrics}")
                
                logger.info("💾 Saving model...")
                model_to_save = self.trainer.model.module if hasattr(self.trainer.model, 'module') else self.trainer.model
                model_to_save.save_pretrained("outputs")
                logger.info("✅ Model saved to ./outputs")
            else:
                logger.info("🐛 Debug mode - skipping training")
                
        except Exception as e:
            logger.error(f"❌ Error during initialization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def evaluate_x_examples(self, x=5):
        if not self.dataset:
            logger.warning("⚠️ No dataset provided for evaluation.")
            return

        logger.info(f"🔍 Evaluating {x} random examples from dataset...")
        
        # Randomly sample from the dataset
        samples = random.sample(self.dataset, min(x, len(self.dataset)))

        for idx, sample in enumerate(samples):
            prompt = sample.get('prompt') or sample.get('input') or ""
            expected = sample.get('response') or sample.get('output') or "N/A"

            logger.info(f"\n🧪 Example {idx + 1}")
            logger.info(f"📝 Input: {prompt}")
            logger.info(f"🎯 Expected: {expected}")

            try:
                generated = self.generate_test_text(prompt=prompt, max_length=500)  
                logger.info(f"🤖 Generated: {generated}")
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")



    def init_trainer(self):
        logger.info("📊 Splitting dataset...")

        split = self.dataset.train_test_split(test_size=0.2, seed=666)
        train_data = split['train']
        eval_data = split['test']

        logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
        
        logger.info("⚙️ Creating trainer...")
        trainer = SFTTrainer(
            model=self.lora_model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=self.init_training_args(),
            peft_config=self.loadPeftConfig(),
            #tokenizer=self.tokenizer,
            #packing=False,  
        )
        return trainer

    def init_training_args(self):
        # Reduced steps for faster debugging
        max_steps = 50 if self.debug_mode else 500
        
        logger.info(f"🎯 Training arguments: max_steps={max_steps}")
        
        training_arguments = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,  # Reduced for memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # Compensate for smaller batch
            optim="paged_adamw_32bit",
            save_steps=25 if self.debug_mode else 100,
            logging_steps=5 if self.debug_mode else 10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=max_steps,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",
            save_total_limit=2,  # Keep only 2 checkpoints
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        return training_arguments

    def loadPeftConfig(self):
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

    def loadBnBConfig(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
    def load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            trust_remote_code=True,
            device_map="auto"  # Automatic device placement
        )
        self.model.config.use_cache = False
        
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_test_text(self, prompt, max_length=100):
        try:
            result = self.pipeline(prompt)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation failed: {str(e)}"
    
    def print_trainable_parameters(self, model, model_name="Model"):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        message = f"{model_name} - Trainable: {trainable_params:,}, Total: {all_param:,}, Trainable%: {100 * trainable_params / all_param:.2f}%"
        logger.info(message)

# Your dataset class (same as before but with logging)
class FineTuningDataset:
    def __init__(self, data):
        self.data = data
        self.traffic_examples = []
        logger.info(f"📂 Initializing dataset with {len(data)} examples")
        self.create_traffic_examples()
        logger.info(f"✅ Created {len(self.traffic_examples)} training examples")
        if len(self.traffic_examples) > 0:
            logger.info(f"Sample example: {self.traffic_examples[0]}")
        self.prompts = []
        
    def create_traffic_examples(self):
        for _, row in self.data.iterrows():
            example = {
                "messages": [
                    {"role": "user", "content": row["user_message"]},
                    {"role": "assistant", "content": row["assistant_message"]}
                ]
            }
            self.traffic_examples.append(example)
        return self.traffic_examples
    
    def parse_input_output(self, example):
        input_text = example["messages"][0]["content"]
        output_text = example["messages"][1]["content"]
        return input_text, output_text
    
    def get_prompt(self, example):
        SYSTEM_PROMPT = """Ti si radijski prometni napovedovalec. Na podlagi surovih vhodnih podatkov o prometu (nesreče, zapore cest, čakalne dobe na mejah, okvare vozil, vreme itd.) pripravi jasen, urejen in tekoč prometni pregled, primeren za radijsko oddajo. 
        Poročaj v formalnem, a razumljivem jeziku. Najprej izpostavi najpomembnejše prometne dogodke (npr. nesreče ali popolne zapore.
        Sledi poročanje po kategorijah (nesreče, okvare vozil, dela na cesti, meje). Ne kopiraj besedila dobesedno preoblikuj podatke v zgoščen in tekoč govor. Vedno uporabi nevtralen ton in poročaj le relevantne informacije."""
        user_msg, assistant_msg = self.parse_input_output(example)
        
        formatted_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        
        self.prompts.append({
            'text': formatted_text
        }) 
        
    def generate_dataset(self):
        logger.info("🔄 Converting to HuggingFace dataset format...")
        for example in self.traffic_examples:
            self.get_prompt(example)
        dataset = Dataset.from_list(self.prompts)
        logger.info(f"✅ Dataset ready: {len(dataset)} examples")
        return dataset


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 STARTING FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    # Check if debug mode
    debug_mode = "--debug" in sys.argv or len(sys.argv) > 1 and sys.argv[1] == "debug"
    
    try:
        logger.info("📁 Loading training data...")
        df = pd.read_csv("ul-fri-nlp-course-project-2024-2025-onjyfans/report/code/trainingdataset2.csv")
        logger.info(f"✅ Loaded {len(df)} rows from CSV")
        
        # Show data info
        logger.info(f"Columns: {list(df.columns)}")
        if 'user_message' in df.columns and 'assistant_message' in df.columns:
            logger.info("✅ Required columns found")
        else:
            logger.error("❌ Missing required columns: user_message, assistant_message")
            sys.exit(1)
        
        # Limit data for debugging
        if debug_mode:
            df = df.head(20)  # Only 20 examples for quick testing
            logger.info(f"🐛 Debug mode: Using only {len(df)} examples")
        
        logger.info("🏗️ Creating dataset...")
        finetuning_dataset = FineTuningDataset(df)
        finetuning_hf_dataset = finetuning_dataset.generate_dataset()
        
        logger.info("🤖 Starting fine-tuner...")
        tuner = FineTuner(
            modelName="cjvt/GaMS-9B-Instruct",  # Start with smaller model
            tokenizerName="cjvt/GaMS-9B-Instruct",
            dataset=finetuning_hf_dataset,
            debug_mode=debug_mode
        )
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📋 Check log file: {log_file}")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
