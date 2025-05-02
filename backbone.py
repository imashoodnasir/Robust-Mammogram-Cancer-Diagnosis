# backbone.py
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def build_backbone(input_shape=(224, 224, 3)):
    base_model = EfficientNetV2S(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True
    inputs = Input(shape=input_shape)
    features = base_model(inputs)
    model = Model(inputs=inputs, outputs=features, name="EfficientNetV2_Backbone")
    return model
