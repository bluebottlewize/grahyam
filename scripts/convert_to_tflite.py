import sys
from tf_keras.models import load_model
from tensorflow import lite

TRAIN_FILE = sys.argv[1]

print('loading from', TRAIN_FILE)

model = load_model(TRAIN_FILE)

converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

# Step 3: Save the TFLite model to a file
with open(TRAIN_FILE + '.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been converted and saved as" + TRAIN_FILE + ".tflite")