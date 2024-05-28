import pandas as pd

# Path to the data file
file_path = 'data/diabetes.txt'

# Read the dataset
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Create pseudo input-response pairs
data = {'input_text': [], 'response_text': []}
for i in range(0, len(lines) - 1):
    input_text = lines[i].strip()
    response_text = lines[i + 1].strip()
    if input_text and response_text:
        data['input_text'].append(input_text)
        data['response_text'].append(response_text)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data/prepared_data.csv', index=False)
