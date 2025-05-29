#!/usr/bin/env python3
"""
Quick test script to validate setup without full training
Run this first to catch issues early!
"""
print("IMPORTING /*")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import sys
import os
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


def test_gpu_availability():
    print("=== GPU Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

def test_data_loading():
    print("=== Data Loading Test ===")
    try:
        # Adjust path as needed
        df = pd.read_csv("../data/RTVSlo/trainingdataset.csv")
        print(f"✅ CSV loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['user_message', 'assistant_message']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✅ Required columns present")
            
        # Show sample data
        print("\nSample rows:")
        for i in range(min(2, len(df))):
            print(f"Row {i}:")
            print(f"  User: {df.iloc[i]['user_message'][:100]}...")
            print(f"  Assistant: {df.iloc[i]['assistant_message'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_loading():
    print("=== Model Loading Test ===")
    try:
        # Test with smaller model first
        model_name = "cjvt/GaMS-1B"  # Start small!
        print(f"Testing with: {model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model.config.use_cache = False
        print("✅ Model loaded")
        
        # Test LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Applying LoRA...")
        lora_model = get_peft_model(model, lora_config)
        print("✅ LoRA applied")
        
        # Print parameter counts
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        print(f"Trainable params: {trainable_params:,}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    print("=== Dataset Creation Test ===")
    try:
        # Create tiny test dataset
        test_data = pd.DataFrame({
            'user_message': ['Hello', 'How are you?'],
            'assistant_message': ['Hi there!', 'I am fine, thanks!']
        })
        
        # Test your dataset class
        from your_script import FineTuningDataset  # Adjust import
        
        dataset = FineTuningDataset(test_data)
        hf_dataset = dataset.generate_dataset()
        
        print(f"✅ Dataset created with {len(hf_dataset)} examples")
        print("Sample example:")
        print(hf_dataset[0])
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Running Quick Setup Tests")
    print("=" * 50)
    
    tests = [
        ("GPU", test_gpu_availability),
        ("Data", test_data_loading),
        ("Model", test_model_loading),
        # ("Dataset", test_dataset_creation),  # Uncomment when ready
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
        print()
    
    print("=" * 50)
    print("📋 Test Summary:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n⚠️  Some tests failed. Fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()