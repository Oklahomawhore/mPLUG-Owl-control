import json

# Load all entries from the JSON file
with open('llava_instruct_150k.json', 'r') as f:
    data = json.load(f)

# Update the "image" entry values
for entry in data:
    if "image" in entry:
        entry["image"] = [f"/data/wangshu/data/train2017/{ entry['image']}"]
    entry["task_type"] = "llava_sft"

    text = """The following is a conversation between a curious human and AI assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions.\n"""
    for speech in entry["conversations"]:
        if speech["from"] == "human":
            text += "Human: "
            text += speech["value"]

        elif speech["from"] == "gpt":
            text += "AI: "   
            text += speech["value"]
    entry["text"] = text
    del entry["conversations"]
    del entry["id"]
# Calculate the split index based on the 9:1 ratio
split_index = int(0.9 * len(data))

# Write the first 90% of updated entries to the train file in JSONL format
with open('sft_v0.1_train.jsonl', 'w') as f:
    for entry in data[:split_index]:
        f.write(json.dumps(entry) + '\n')

# Write the remaining 10% of updated entries to the dev file in JSONL format
with open('sft_v0.1_dev.jsonl', 'w') as f:
    for entry in data[split_index:]:
        f.write(json.dumps(entry) + '\n')
