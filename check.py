import pandas as pd
import json
# Load the two files
file1_path = 'data.csv'
file2_path = 'final_data.csv'

# Read the CSV files
data1 = pd.read_csv(file1_path)
data2 = pd.read_csv(file2_path)
print(data2.keys)

# Check if they are the same
are_files_same = data1.equals(data2)

print("Are the files the same?", are_files_same)

# dat = pd.read_csv('temp.csv')
# x=dat.head(1)
# # Convert the DataFrame to a dictionary
# x_dict = x.to_dict(orient='records')

# # Convert the dictionary to a JSON string
# var = json.dumps(x_dict, indent=4)

# # Print the JSON string
# print(var)
