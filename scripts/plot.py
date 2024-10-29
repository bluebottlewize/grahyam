import matplotlib.pyplot as plt

# Define the coordinates
coords = open("../predict.txt", "r").readlines()

coordinates = []

for i in coords:
    x = i.split()
    x = (int(x[0]), int(x[1]))
    coordinates.append(x)

print(coordinates)

# Split the coordinates into x and y values
x, y = zip(*coordinates)

# Create the scatter plot
plt.scatter(x, y)

plt.plot(x, y, color='orange', linestyle='-', marker='o')

plt.xlim(min(x) - 10, max(x) + 10)
plt.ylim(min(y) - 10, max(y) + 10)

# plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

# Add titles and labels
plt.title('Scatter Plot of Coordinates')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show grid
plt.grid()

# Display the plot
plt.show()

