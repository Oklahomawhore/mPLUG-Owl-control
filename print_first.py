# Open the JSONL file and read the first line
with open('sft_v0.1_train.jsonl', 'r') as f:
    first_line = f.readline()

# Print the first line
print(first_line)