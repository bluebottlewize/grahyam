import alphabets

# Get all variables defined in the module
variables = [var for var in dir(alphabets) if not var.startswith('__')]

# Print all variable names and their values
for var in variables:
    print(f"{var}: {getattr(alphabets, var)}")
