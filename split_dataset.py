# Load all lines from the JSONL file
with open('sft_v0.1_train.jsonl', 'r') as f:
    lines = f.readlines()

# Calculate the split index based on the 9:1 ratio
split_index = int(0.9 * len(lines))

# Write the first 90% of lines back to the train file
with open('sft_v0.1_train.jsonl', 'w') as f:
    f.writelines(lines[:split_index])

# Write the remaining 10% of lines to the dev file
with open('sft_v0.1_dev.jsonl', 'w') as f:
    f.writelines(lines[split_index:])
