import tensorflow as tf

model_path = "C:/Users/Mohanvel/Downloads/gender_detection_model.h5"
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = "C:/Users/Mohanvel/Downloads/gender_detection.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format!")
print(f"Saved to: {tflite_model_path}")
