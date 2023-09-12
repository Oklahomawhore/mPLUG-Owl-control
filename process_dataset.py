import json
from tqdm import tqdm
# Read the JSON file
with open('llava_instruct_150k.json', 'r') as f:
    data = json.load(f)


# Open a file to write the transformed data in JSONL format
with open('sft_v0.1_train.jsonl', 'w') as f:
    for entry in tqdm(data, desc="generating train"):
        # Extract the necessary fields
        id_ = entry['id']
        conversations = entry['conversations']
        image = entry['image']

        # Construct the new format
        if image:
            transformed_entry = {
                "image": [image],
                "text": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            }
            for conversation in conversations:
                transformed_entry["text"] += f"{conversation['from'].capitalize()}: {conversation['value']}\n"
            transformed_entry["task_type"] = "llava_sft"
        else:
            transformed_entry = {
                "text": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            }
            for conversation in conversations:
                transformed_entry["text"] += f"{conversation['from'].capitalize()}: {conversation['value']}\n"
            transformed_entry["task_type"] = "gpt4instruct_sft"

        # Write the transformed entry to the JSONL file
        f.write(json.dumps(transformed_entry) + '\n')
