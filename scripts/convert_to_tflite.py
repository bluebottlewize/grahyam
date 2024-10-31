from tf_keras.models import load_model
from tensorflow import lite
# Step 1: Load the model from the HDF5 file
model = load_model('../models/grahyam_v3.h5')

# Step 2: Convert the model to TFLite format


converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

# Step 3: Save the TFLite model to a file
with open('../models/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been converted and saved as model.tflite")
