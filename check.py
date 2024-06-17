import pandas as pd

# Load the two files
file1_path = 'data.csv'
file2_path = 'final_data.csv'

# Read the CSV files
data1 = pd.read_csv(file1_path)
data2 = pd.read_csv(file2_path)

# Check if they are the same
are_files_same = data1.equals(data2)

print("Are the files the same?", are_files_same)
