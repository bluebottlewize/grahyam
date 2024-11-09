import numpy as np
from tf_keras.models import load_model

label_enum = ['ക', 'ഖ', 'ഗ', 'ഘ', 'ങ']

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



model = load_model('./grahyam_v1.h5')

coords = open("./predict.txt", "r").readlines()

coordinates = []
for i in coords:
    xy = i.split()
    coordinates.append((int(xy[0]), int(xy[1])))

normalized_coordinates = normalize_coordinates(coordinates)

sequences = [coordinates]
np_normalized_sequences = np.asarray(sequences)

Xpad = np.full((len(np_normalized_sequences), 137, 2), fill_value=-10)
for s, x in enumerate(np_normalized_sequences):
    seq_len = np.shape(x)[0]
    Xpad[s, 0:seq_len, :] = x

print(Xpad)

predictions = model.predict(Xpad)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", label_enum[predicted_classes[0]])
