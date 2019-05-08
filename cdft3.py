#####
# MobileNetV2 Monkey Species Fine Tuning with dataset optimization
####
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import json

DATASET_PATH = './data'
IMAGE_SIZE = (160, 160)
NUM_CLASSES = 2
BATCH_SIZE = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS = 60
WEIGHTS_FINAL = 'model-resnet50-final.h5'

pretrain_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
pretrain_batches = pretrain_datagen.flow_from_directory(DATASET_PATH + '/training',
                                                        target_size=IMAGE_SIZE,
                                                        interpolation='bicubic',
                                                        class_mode='binary',
                                                        shuffle=True,
                                                        batch_size=251)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='binary',
                                                  shuffle=False,
                                                  batch_size=32)
# show class indices
print('****************')
for cls, idx in pretrain_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

base_learning_rate = 0.0001
eval_model = MobileNetV2(weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
eval_model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                   metrics=['accuracy', 'mean_squared_error'])

with open('monkeylabel.json') as json_file:
    monkeydict = json.load(json_file)

helpful_data = list()
helpful_labels = list()
harmful_data = list()
harmful_predicted_labels = list()

count = 0
for x, y in pretrain_batches:
    for i, samp in enumerate(x):
        pred = eval_model.predict(np.array([samp]), verbose=1)
        top_indices = pred[0].argsort()[-3:][::-1][0]
        is_related = str(top_indices) in monkeydict
        if not is_related:
            top_pred = decode_predictions(pred, top=3)[0][0]
            print('Predicted:', top_pred)
            harmful_data.append(samp)
            harmful_predicted_labels.append(top_pred)
        else:
            helpful_data.append(samp)
            helpful_labels.append(y[i])
        count += 1
        print(count)
    if count > 251:
        break

helpful_data = np.array(helpful_data)
helpful_labels = np.array(helpful_labels)
harmful_data = np.array(harmful_data)
harmful_predicted_labels = np.array(harmful_predicted_labels)
#print(helpful_data)
#print(helpful_labels)
#print(harmful_data)
#print(harmful_predicted_labels)
train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow(helpful_data, helpful_labels)

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = True

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(1, name='prediction_layer'))
model.add(Dense(1, activation="sigmoid"))

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(str(200 // BATCH_SIZE))
hist = model.fit_generator(train_batches,
                           validation_data=valid_batches,
                           epochs=NUM_EPOCHS,
                           steps_per_epoch=200 // BATCH_SIZE,
                           validation_steps=valid_batches.samples // BATCH_SIZE)
model.save('cdft_mobilenet_optimized1.h5')
print(hist.history)
