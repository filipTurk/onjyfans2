import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import logging
import sys
from datetime import datetime
import time
from peft import PeftModel # <--- IMPORT THIS

# --- Setup Logging (Same as your training script for consistency) ---
def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"prediction_log_{timestamp}.log"
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

logger, log_file = setup_logging()

class ModelInference:
    def __init__(self, base_model_path: str, adapter_path: str): # <--- MODIFIED
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path # <--- NEW
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        logger.info(f"🚀 Initializing Inference for base model: {self.base_model_path}")
        if adapter_path:
            logger.info(f"    with adapter: {self.adapter_path}")
        start_time = time.time()
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # --- STEP 1: LOAD THE BASE MODEL ---
            logger.info(f"🤖 Loading base model: {self.base_model_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
            base_model.config.use_cache = True

            # --- STEP 2: LOAD THE TOKENIZER (from the base model) ---
            logger.info(f"🔤 Loading tokenizer from {self.base_model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # --- STEP 3: LOAD AND APPLY THE LoRA ADAPTER ---
            logger.info(f"🔧 Applying LoRA adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            logger.info("✅ Adapter applied successfully.")
            
            # --- STEP 4: CREATE THE PIPELINE with the combined model ---
            logger.info("🛠️ Creating text generation pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            logger.info(f"✅ Model ready in {time.time() - start_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise

    def format_prompt(self, input_message: str, input_json: str) -> str:
        """
        Creates the full prompt in the specific format the model was trained on.
        The prompt must end with '<|assistant|>\n' to signal the model to start generating.
        """
        SYSTEM_PROMPT = """Deluješ kot poročevalec prometnih informacij za radijsko postajo. Na podlagi vhodnih podatkov generiraj kratko in neposredno besedilo v katerem povzameš le najpomembnejše informacije."""
        
        user_message = f"""
**Vhodno sporočilo:**
{input_message}
**Vhodni JSON podatki:**
{input_json}
"""
        # This format MUST match the training format
        formatted_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_message.strip()}\n<|assistant|>\n"
        return formatted_prompt

    def generate(self, prompt: str, max_new_tokens: int = 350) -> str:
        """
        Generates text using the loaded pipeline.
        
        Args:
            prompt (str): The full, formatted prompt.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text from the assistant.
        """
        if not self.pipeline:
            logger.error("Pipeline is not initialized.")
            return "Error: Pipeline not initialized."

        try:
            sequences = self.pipeline(
                prompt,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                # Add pad_token_id to suppress warnings
                pad_token_id=self.tokenizer.pad_token_id 
            )
            
            # Extract only the newly generated text (the assistant's response)
            full_text = sequences[0]['generated_text']
            # The response is everything after the final '<|assistant|>\n'
            assistant_response = full_text.split("<|assistant|>\n")[-1]

            return assistant_response.strip()
        
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return f"Error during generation: {e}"

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 STARTING INFERENCE SCRIPT")
    logger.info("=" * 60)

    # --- NEW Configuration ---
    # The original model you fine-tuned FROM
    BASE_MODEL_PATH = "cjvt/GaMS-9B-Instruct" 
    # The path where your adapter was saved TO
    ADAPTER_PATH = "./outputs" 
    
    DATA_PATH = "onjyfans2/report/code/trainingdataset_optimized_normalized.csv"
    NUM_EXAMPLES_TO_TEST = 35
    OUTPUT_CSV_PATH = "predictions_output_gams_prompting002.csv"

    try:
        # --- MODIFIED: Pass both paths to the constructor ---
        inference_engine = ModelInference(
            base_model_path=BASE_MODEL_PATH, 
            adapter_path=ADAPTER_PATH
        )

        # 2. Load the dataset
        logger.info(f"📁 Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        df_sample = df.sample(n=NUM_EXAMPLES_TO_TEST, random_state=42)  # Randomly sample the specified number of examples
        logger.info(f"✅ Loaded {len(df)} rows, will test on the first {len(df_sample)}.")

        output_df = df_sample.copy()  # Keep a copy for output if needed
        
        # 3. Loop through examples and generate predictions
        for index, row in df_sample.iterrows():
            logger.info("\n" + "="*20 + f" Example {index + 1} " + "="*20)

            # Get data from the row
            input_msg = row["Input_Message"]
            input_json = row["Input_JSON"]
            expected_output = row["Output_Message"]

            # Format the prompt
            prompt = inference_engine.format_prompt(input_msg, input_json)
            
            logger.info(f"▶️ PROMPT (sent to model):\n{prompt}")

            # Generate the prediction
            start_gen_time = time.time()
            generated_output = inference_engine.generate(prompt)
            gen_time = time.time() - start_gen_time
            
            logger.info(f"✅ MODEL OUTPUT (in {gen_time:.2f}s):\n{generated_output}")
            logger.info(f"🎯 EXPECTED OUTPUT:\n{expected_output}")
            
            output_df.at[index, "Generated_Output"] = generated_output

        logger.info("\n" + "="*60)
        logger.info("🎉 Inference script finished successfully!")
        logger.info(f"📋 Check the full log at: {log_file}")
        output_df.to_csv("predictions_output_gams_prompting.csv", index=False)

    except Exception as e:
        logger.error(f"💥 A critical error occurred in the main script: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)