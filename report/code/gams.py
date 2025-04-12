import transformers

model_id = "cjvt/OPT_GaMS-1B-Chat"
pipeline = transformers.pipeline("text-generation", model=model_id, device_map="auto")

prompt = "kaj je dežela?"

message = [{"role": "user", "content": f"{prompt}"}]
response = pipeline(message, max_length=100, )
print("Model's response:", response[0]["generated_text"][1]["content"])