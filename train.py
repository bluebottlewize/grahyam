import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras
from tf_keras import Sequential
from tf_keras.utils import to_categorical
from tf_keras.utils import Sequence
from tf_keras.layers import LSTM, Dense, Masking, Input, InputLayer, Dropout
import numpy as np
from rdp import rdp

initial_data = {}
test_data = {}

def read_files_in_directory(directory):
    coord_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            coords = []
            file_path = os.path.join(root, file)
            # Check if it's a text file (you can adjust this)
            if file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    content = f.readlines()
                    for i in content:
                        xy = i.split()
                        coords.append((int(xy[0]), int(xy[1])))
            coord_list.append(coords)

            initial_data[directory[-1::]] = coord_list

def load_train_data(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"Directory: {dir_path}")

            coord_list = []
            for root1, dirs1, files1 in os.walk(dir_path):
                for file in files1:
                    coords = []
                    file_path = os.path.join(root1, file)
                    # Check if it's a text file (you can adjust this)
                    if file.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            content = f.readlines()
                            for i in content:
                                xy = i.split()
                                coords.append((int(xy[0]), int(xy[1])))
                    coord_list.append(coords)
                    coord_list.append(coords[::-1])

            print(dir_path[dir_path.index('/', 4) + 1:])
            initial_data[dir_path[dir_path.index('/', 4) + 1:]] = coord_list


def load_test_data(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"Directory: {dir_path}")

            coord_list = []
            for root1, dirs1, files1 in os.walk(dir_path):
                for file in files1:
                    coords = []
                    file_path = os.path.join(root1, file)
                    # Check if it's a text file (you can adjust this)
                    if file.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            content = f.readlines()
                            for i in content:
                                xy = i.split()
                                coords.append((int(xy[0]), int(xy[1])))
                    coord_list.append(coords)

            test_data[dir_path[dir_path.index('/', 4) + 1:]] = coord_list


load_train_data('./train')
load_test_data('./test')

print(test_data)


def plot_letters(coordinates_list):

    x_list = []
    y_list = []

    for c in coordinates_list:
        x, y = zip(*c)
        x_list.append(x)
        y_list.append(y)

    plt.xlim(min(x_list[0]), max(x_list[0]))
    plt.ylim(min(y_list[0]), max(y_list[0]))

    for x, y in zip(x_list, y_list):
        color = (random.random(), random.random(), random.random())
        plt.scatter(x, y, color=color, s=10)  # First set in blue
        plt.plot(x, y, color=color, linestyle='-', marker='o', markersize=1)

        plt.xlim(min(min(x), plt.xlim()[0] + 10) - 10, max(max(x), plt.xlim()[1] - 10) + 10)
        plt.ylim(min(min(y), plt.ylim()[0] + 10) - 10, max(max(y), plt.ylim()[1] - 10) + 10)

    # Invert y-axis if needed
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # Add titles and labels
    plt.title('Scatter Plot of Multiple Coordinate Sets')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show grid
    plt.grid()

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

def normalize_coordinates(coordinates):
    x_list, y_list = zip(*coordinates)
    x_min = min(x_list)
    y_min = min(y_list)

    x_max = max(x_list)
    y_max = max(y_list)

    n_range = 100

    x_range = x_max - x_min
    y_range = y_max - y_min

    n_factor_x = x_range / n_range
    n_factor_y = y_range / n_range

    n_factor = n_factor_x
    
    if y_range > x_range:
        n_factor = n_factor_y

    n_min_x = int(x_min / n_factor)
    n_min_y = int(y_min / n_factor)

    normalized_coordinates = []

    for c in coordinates:
        p = (int(c[0] / n_factor) - n_min_x, int(c[1] / n_factor) - n_min_y)
        normalized_coordinates.append(p)

    return normalized_coordinates

normalized_coordinates_list = []
normalized_reduced_coordinates_list = []

for c in initial_data['ഘ'][30:60]:
    normalized_coordinates_list.append(normalize_coordinates(c))
    normalized_reduced_coordinates_list.append(rdp(np.array(normalized_coordinates_list[-1]), epsilon=1))

# label_enum = ['ക', 'ഖ', 'ഗ', 'ഘ', 'ങ']
label_enum = ['അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഒ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'സ്സ', '\u0D3E', '\u0D3F', '\u0D40', '\u0D41', '\u0D42', '\u0D43', '\u0D46', '\u0D47', '\u0D57', '\u0D4D']

sequences = []
labels = []
test_sequences = []
test_labels = []

for i in initial_data.keys():
    sequences += initial_data[i]
    test_sequences += test_data[i]
    labels += [label_enum.index(i)] * len(initial_data[i])
    test_labels += [label_enum.index(i)] * len(test_data[i])

print(test_labels)

normalized_sequences = []
normalized_test_sequences = []
normalized_reduced_sequences = []
normalized_reduced_test_sequences = []

for coordinates in sequences:
    normalized_sequences.append(normalize_coordinates(coordinates))
    # normalized_reduced_sequences.append(rdp(np.array(normalized_sequences[-1]), epsilon=2))
    normalized_reduced_sequences.append([tuple(i) for i in rdp(np.array(normalized_sequences[-1]), epsilon=1)])

for coordinates in test_sequences:
    normalized_test_sequences.append(normalize_coordinates(coordinates))
    print(normalized_test_sequences[-1])
    normalized_reduced_test_sequences.append([tuple(i) for i in rdp(np.array(normalized_test_sequences[-1]), epsilon=1)])
    print(normalized_reduced_test_sequences[-1])

# print("normal", normalized_sequences)
# print("normal reduced", normalized_reduced_sequences)

# normalized_sequences = normalized_reduced_sequences
# normalized_test_sequences = normalized_reduced_test_sequences


c = list(zip(labels, normalized_sequences))
random.shuffle(c)
labels, normalized_sequences = zip(*c)

c = list(zip(test_labels, normalized_test_sequences))
random.shuffle(c)
test_labels, normalized_test_sequences = zip(*c)

seq_lens = [len(i) for i in normalized_sequences]
dimension = 2
lstm_units = 32
print(seq_lens)    

special_value = -10
max_seq_len = max(seq_lens) + 50

pad = (-10, -10)

for c in normalized_sequences:
    c += [pad] * (max_seq_len - len(c))

for c in normalized_test_sequences:
    c += [pad] * (max_seq_len - len(c))

np_normalized_sequences = np.asarray(normalized_sequences)
np_normalized_test_sequences = np.asarray(normalized_test_sequences)

# print(np_normalized_sequences)

Xpad = np.full((len(np_normalized_sequences), max_seq_len, dimension), fill_value=special_value)
for s, x in enumerate(np_normalized_sequences):
    seq_len = np.shape(x)[0]
    Xpad[s, 0:seq_len, :] = x

print(Xpad)

Xpad_test = np.full((len(np_normalized_test_sequences), max_seq_len, dimension), fill_value=special_value)
for s, x in enumerate(np_normalized_test_sequences):
    seq_len = np.shape(x)[0]
    Xpad_test[s, 0:seq_len, :] = x

print(Xpad_test)

model2 = Sequential()

num_letters = len(label_enum)

y_encoded = to_categorical(labels, num_classes=num_letters)

print(max_seq_len)
print(len(Xpad))

# input_layer = Input(shape=(max_seq_len, dimension))
# masking_layer = Masking(mask_value=special_value)(input_layer)
# model2.add(InputLayer(input_shape=(max_seq_len, dimension)))
# model2.add(Masking(mask_value=special_value))
model2.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
# model2.add(masking_layer)
model2.add(LSTM(64, return_sequences=True))
model2.add(Dropout(0.5))
model2.add(LSTM(64, return_sequences=True))
model2.add(LSTM(64))
model2.add(Dense(num_letters, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model2.summary())
model2.fit(Xpad, y_encoded, epochs=40, batch_size=32, validation_split=0.2)

predictions = model2.predict(Xpad_test)

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted classes
print("Predicted classes:", predicted_classes)
print(test_labels)

total = len(test_labels)
correct = 0

for i, j in enumerate(test_labels):
    if predicted_classes[i] == j:
        correct += 1
    else:
        print("correct: ", label_enum[j], j, " predicted: ", label_enum[predicted_classes[i]], predicted_classes[i])

print(correct, " / ", total)

model2.save('./models/grahyam_v6_extra_layer.h5')

# converter = tf.lite.TFLiteConverter.from_keras_model(model2)
# tflite_model = converter.convert()

# Save the TFLite model to a file
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

print("Model has been converted and saved as model.tflite")
