import os
from tensorflow import keras
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, load_quantize=False):
        self.quantize = load_quantize
        keras.backend.set_learning_phase(0)
        self.__load_model(load_quantize=load_quantize)
    
    def __load_model(self, dir="model_4", load_quantize=True):
        with open(os.path.join(dir, "model.json"), "r") as f:        
            loaded_model_json = f.read()
        
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(os.path.join(dir, "model.h5"))
        if load_quantize:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            tflite_quant_model = converter.convert()
            open("converted_quant_model.tflite", "wb").write(tflite_quant_model)
            interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
            interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            
            self.model = interpreter
        else:
            print(tf.config.experimental.list_physical_devices('GPU'))
        
    def __compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=["accuracy"])
    
    def predict(self, X_train):
        if self.quantize:
            out = []
            for i in range(len(X_train)):
                roi = np.array([X_train[i]])
                self.model.set_tensor(self.input_details[0]['index'], roi)
                self.model.invoke()
                output_data = self.model.get_tensor(self.output_details[0]['index'])
                out.append(np.argmax(output_data))
            return out
        pred = self.model.predict(X_train, workers=2)
        self.pred = np.argmax(pred, axis=1)
        return self.pred
        