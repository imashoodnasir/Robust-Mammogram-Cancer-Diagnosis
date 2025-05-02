# train.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_image
from augment import get_augmentation_pipeline
from backbone import build_backbone
from cbam import cbam_block
from classifier import build_classifier

def build_full_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    augmented = get_augmentation_pipeline()(inputs)
    
    backbone = build_backbone(input_shape)
    features = backbone(augmented)
    attention_features = cbam_block(features)
    outputs = build_classifier(attention_features)

    model = Model(inputs, outputs, name="EfficientNetV2_CBAM_Classifier")
    return model

def compile_and_train(model, train_ds, val_ds, epochs=50):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr]
    )
    return model, history
