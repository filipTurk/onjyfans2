import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import random
from datasets import load_dataset, Dataset
import json
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

class FineTuner:
    def __init__(self, modelName="cjvt/GaMS-27B", tokenizerName="cjvt/GaMS-27B", dataset=None):
        self.model_name = modelName
        self.tokenizer_name = tokenizerName
        self.dataset = dataset  # FIX: Store the dataset
        
        self.bnb_config = self.loadBnBConfig()
        self.load_model(model_name=modelName)
        self.load_tokenizer()
        
        # FIX: Create pipeline with base model first
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.generate_test_text(prompt="Hello, how are you?", max_length=50)
        
        self.max_seq_length = 512
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_r = 16
        
        self.lora_model = get_peft_model(
            self.model,
            self.loadPeftConfig()
        )

        self.print_trainable_parameters(self.model)
        print("Model and tokenizer loaded successfully.")
        self.print_trainable_parameters(self.lora_model)
        
        # Update pipeline to use LoRA model
        self.pipeline = pipeline("text-generation", model=self.lora_model, tokenizer=self.tokenizer)
        self.generate_test_text(prompt="Hello, how are you?", max_length=50)
        
        print("LoRA model initialized successfully.")
        self.trainer = self.init_trainer()
        print("Trainer initialized successfully.")
        
        print("Training started.")
        self.trainer.train()
        print("Training completed.")
        metrics = self.trainer.evaluate()
        print(metrics)

        model_to_save = self.trainer.model.module if hasattr(self.trainer.model, 'module') else self.trainer.model
        model_to_save.save_pretrained("outputs")

    def init_trainer(self):
        train_data, eval_data = train_test_split(self.dataset, test_size=0.2, random_state=666)
        # FIX: Remove recursive call
        trainer = SFTTrainer(
            model=self.lora_model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=self.init_training_args(),
            peft_config=self.loadPeftConfig(),
            tokenizer=self.tokenizer,
            packing=False,  
            max_seq_length=self.max_seq_length,
        )
        return trainer

    def init_training_args(self, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=2, per_device_eval_batch_size=2):
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,  # FIX: Add this parameter
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=500,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",
            evaluation_strategy="steps",  # FIX: Add evaluation strategy
            eval_steps=100,  # FIX: Add eval steps
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
            trust_remote_code=True
        )
        self.model.config.use_cache = False
        
    def load_tokenizer(self):
        # FIX: Use correct attribute name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_pipeline(self, model=None):
        return pipeline("text-generation", model=self.lora_model, tokenizer=self.tokenizer)

    def generate_test_text(self, prompt, max_length=100):
        return self.pipeline(prompt, max_length=max_length, do_sample=True, temperature=0.7)[0]['generated_text']
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params}, all params: {all_param}, trainable%: {100 * trainable_params / all_param:.2f}")

class FineTuningDataset:
    def __init__(self, data):
        self.data = data
        self.traffic_examples = []
        self.create_traffic_examples()
        print(f"Dataset initialized with {len(self.data)} examples.")
        print(f"5 examples: {self.traffic_examples[:5]}")
        self.prompts = []
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

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
    
    def get_random_example(self):
        return random.choice(self.traffic_examples)
    
    def parse_input_output(self, example):
        input_text = example["messages"][0]["content"]
        output_text = example["messages"][1]["content"]
        return input_text, output_text
    
    def get_next_example(self, current_index):
        next_index = current_index + 1
        if next_index < len(self.traffic_examples):
            return self.parse_input_output(self.traffic_examples[next_index])
        else:
            return None
    
    def get_prompt(self, example):
        SYSTEM_PROMPT = "You are a helpful assistant that provides traffic information based on user queries."
        user_msg, assistant_msg = self.parse_input_output(example)
        
        # FIX: Format for SFTTrainer - use 'text' field instead of separate prompt/answer
        formatted_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        
        self.prompts.append({
            'text': formatted_text
        }) 
        
    def generate_dataset(self):
        for example in self.traffic_examples:
            self.get_prompt(example)
        return Dataset.from_list(self.prompts)

if __name__ == "__main__":
    df = pd.read_csv("../data/RTVSlo/trainingdataset.csv")
    finetuning_dataset = FineTuningDataset(df)
    finetuning_hf_dataset = finetuning_dataset.generate_dataset()

    tuner = FineTuner(
        modelName="cjvt/GaMS-27B",
        tokenizerName="cjvt/GaMS-27B",
        dataset=finetuning_hf_dataset
    )