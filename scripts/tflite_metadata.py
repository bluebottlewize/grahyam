import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='../models/model.tflite')

# Allocate tensors
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()

model_info = interpreter.get_model()
print("Model version:", model_info.version)
print("Model signature:", model_info.signature)

# Print input shape and data type
for input_detail in input_details:
    print("Input Name:", input_detail['name'])
    print("Input Shape:", input_detail['shape'])
    print("Input Data Type:", input_detail['dtype'])

