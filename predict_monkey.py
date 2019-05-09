import sys
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

def decode_pred(pred):
    labels = [
        'mantled_howler',
        'patas_monkey',
        'bald_uakari',
        'japanese_macaque',
        'pygmy_marmoset',
        'white_headed_capuchin',
        'silvery_marmoset',
        'common_squirrel_monkey',
        'black_headed_night_monkey',
        'nilgiri_langur'
    ]
    top_index = np.argmax(pred)
    return labels[top_index]

relative_image_path = sys.argv[1]
model = tf.keras.models.load_model('monkey_mobilenet_optimized1.h5')
img = image.load_img(relative_image_path, target_size=(160, 160))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted monkey species: %s' % decode_pred(preds))