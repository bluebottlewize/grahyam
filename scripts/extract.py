# Specify the file name
file_name = "alphabets.py"  # Replace with the actual file name

# List to store variable names
variables = []

# Open the file and read line by line
with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip whitespace and ignore empty lines or comment lines
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Split the line on '=' and get the part before '='
        if '=' in line:
            variable = line.split('=')[0].strip()
            variables.append(variable)

# Print the extracted variable names
print("Extracted Variables:")
for var in variables:
    print(var)

# If you want, you can save the variables to a list
print("Variables as a list:")
print(variables)

print("Variables as a list:")
print(", ".join(variables))

