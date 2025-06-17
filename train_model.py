import os
import numpy as np
import tensorflow as tf
from datapreprocess import load_dataset, data_generator, augment
from model_build import build_unet
from utils import get_callbacks

# Cấu hình
data_path = "E:/brain_tumor_segmentation/dataset"
input_shape = (256, 256, 3)
batch_size = 2
epochs_stage1 = 15
epochs_stage2 = 35

# Load data
(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_dataset(data_path)

# Build model
model = build_unet(input_shape=input_shape)

# Loss và Dice Coefficient
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# ------------------ STAGE 1: Freeze encoder (fine-tune decoder only) ------------------
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) and not any(x in layer.name.lower() for x in ['decoder', 'up', 'concat']):
        layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])

callbacks = get_callbacks(checkpoint_path='model_best.h5', monitor='val_loss')

train_gen = data_generator(train_x, train_y, batch_size=batch_size, augment_fn=augment)
val_gen = data_generator(val_x, val_y, batch_size=batch_size, augment_fn=None)

print("Stage 1: Fine-tuning decoder only")
model.fit(
    train_gen,
    epochs=epochs_stage1,
    steps_per_epoch=len(train_x) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_x) // batch_size,
    callbacks=callbacks
)

# ------------------ STAGE 2: Unfreeze all layers ------------------
for layer in model.layers:
    layer.trainable = True

# Compile lại với LR thấp hơn để fine-tune ổn định
fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=fine_tune_optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])

print("Stage 2: Fine-tuning all layers")
model.fit(
    train_gen,
    epochs=epochs_stage2,
    steps_per_epoch=len(train_x) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_x) // batch_size,
    callbacks=callbacks
)
