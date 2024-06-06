from transformers import TableTransformerModel
import json

# Load the pre-trained model
model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

# Load the JSON data
with open('mac_v4.json', encoding="utf8") as f:
    data = json.load(f)

# Function to clean and validate keys
def clean_data(item):
    cleaned_item = {}
    for key, val in item.items():
        # Replace empty values with '0'
        if val == '':
            val = '0'
        # Ensure key is not empty and does not contain invalid characters
        new_key = key.strip()
        if new_key:
            cleaned_item[new_key] = val
    return cleaned_item

# Process and clean each item in the data
cleaned_data = [clean_data(item) for item in data if isinstance(item, dict)]

# Ensure data matches model input requirements
for item in cleaned_data:
    # Prepare the input format as per the model's requirements
    # Note: Adjust this part based on the actual expected input structure of the model
    inputs = {
        'input_ids': item.get('input_ids'),  # Example key
        'attention_mask': item.get('attention_mask')  # Example key
    }
    
    # Remove keys with None values
    inputs = {k: v for k, v in inputs.items() if v is not None}
    
    # Get outputs from the model
    try:
        outputs = model(**inputs)
        print(outputs)
    except Exception as e:
        print(f"Error processing item: {item} with error: {e}")
