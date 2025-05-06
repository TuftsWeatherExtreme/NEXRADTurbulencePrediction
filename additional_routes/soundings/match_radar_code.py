import pandas as pd
import re

# Read the file
with open('soundings_key.txt', 'r') as file:
    lines = file.readlines()

# Prepare empty lists to store the extracted codes and numbers
codes = []
numbers = []

# Regex pattern to extract the number and last code before the last parentheses
pattern = re.compile(r"javascript:g\('(\d+?)'\).*?title=\".*?\((.*?)\)")

# Loop through each line to extract the data
for line in lines:
    match = pattern.search(line)
    if match:
        number = match.group(1)
        
        # Find the last code in parentheses
        title_part = match.group(2)
        code = title_part.split()[-1]  # Take the last part after split

        # Append the extracted values to the lists
        numbers.append(number)
        codes.append(code)

# Create a DataFrame from the lists
df = pd.DataFrame({
    'code': codes,
    'number': numbers
})

# Save the dataframe to a CSV
df.to_csv('radar_codes.csv', index=False)
print(df)
