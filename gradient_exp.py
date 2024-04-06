#%%
import os
from gradientai import Gradient

#%%
gradient_token = os.environ.get("GRADIENT_ACCESS_TOKEN")
workspace = os.environ.get("GRADIENT_WORKSPACE_ID")

#%%
gradient = Gradient(access_token=gradient_token, workspace_id=workspace)
baseModel = gradient.get_base_model(base_model_slug="llama2-7b-chat")

# new_mode_adapter is the copy of the llm we are training
new_mode_adapter = baseModel.create_model_adapter(name="test model")

print(f"Model Adapter with {new_mode_adapter.id}")

sample_query = "### Instruction: Who is Jay Huang \n\n### Response:"
print(f"Asking {sample_query}")

completion = new_mode_adapter.complete(query=sample_query, max_generated_token_count=100, temperature=0).generated_output
print(f"Generated before fine-tune: {completion}")

samples = [
    { "inputs": "### Instruction: Who is Jay Huang? \n\n### Response: Jay Huang works for Ensco" },
    { "inputs": "### Instruction: Who is the person named Jay Huang ? \n\n### Response: Jay Huang is learning AI from youtude videos" },
    { "inputs": "### Instruction: Can you tell me about Jay Huang? \n\n### Response: Jay Huang is a system engineer" },
    { "inputs": "### Instruction: What is Jay Huang's occupation? \n\n### Response: Jay Huang is employed as a system engineer at Ensco." },
    { "inputs": "### Instruction: Provide information on Jay Huang's job role. \n\n### Response: Jay Huang holds the position of a system engineer and works at Ensco." },
    { "inputs": "### Instruction: Describe Jay Huang's role at Ensco. \n\n### Response: Jay Huang's role at Ensco is that of a system engineer." },
    { "inputs": "### Instruction: Who is Jay Huang and where does he work? \n\n### Response: Jay Huang is a system engineer employed at Ensco." },
    { "inputs": "### Instruction: Could you elaborate on Jay Huang's professional background? \n\n### Response: Jay Huang is a system engineer, currently working at Ensco." }
]

num_epochs = 10 #needs more epochs to get better answer
count = 0

while count < num_epochs:
    print(f"Fine-tuning the model, iteration {count+1}")
    new_mode_adapter.fine_tune(samples=samples)
    count += 1

completion = new_mode_adapter.complete(query=sample_query, max_generated_token_count=100, temperature=0).generated_output
print(f"Generate_after fine=tune: {completion}")

ccompletion = new_mode_adapter.complete(query="Who does Jay Huang work for?", max_generated_token_count=100, temperature=0).generated_output 

print(f"{completion}")

#new_mode_adapter.delete()


# %%
filepath =r'C:\Users\tonyh\Desktop\LLM projects\Data\Papers\Dynamic Retrieval Augmented Generation.pdf'

result = gradient.extract_pdf(
    filepath=filepath
)
# %%
text = result['text']