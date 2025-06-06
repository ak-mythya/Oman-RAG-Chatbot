import json

# Load JSON data from file
with open('arabic_chunks_old.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Move chunk_id to metadata for each dictionary
for item in data:
    if 'chunk_id' in item:
        item['metadata']['chunk_id'] = item.pop('chunk_id')

# Save the updated data to a new JSON file
with open('arabic_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)