import os
import numpy as np
import tensorflow as tf

from data_generator import DataGenerator
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

load_dotenv()
test_data_gen = DataGenerator(os.getenv('TEST_DATA_DIR'))
test_data_gen.preproc_func = tf.keras.applications.xception.preprocess_input
test_data_gen.summary()

model = tf.keras.models.load_model(
    '../xception_model.keras', custom_objects=None, compile=True, safe_mode=True
)
model.summary()


progbar = tf.keras.utils.Progbar(len(test_data_gen))

test_pred = np.empty((len(test_data_gen),) + (test_data_gen.batch_size,) + (2,))
test_true = np.empty((len(test_data_gen),) + (test_data_gen.batch_size,) + (2,))

for i in range(len(test_data_gen)-1):
    progbar.update(i)
    x_batch, y_batch = test_data_gen.__getitem__(i)
    test_pred[i] = model(x_batch)
    test_true[i] = y_batch

test_pred = np.concatenate(test_pred)
test_true = np.concatenate(test_true)
assert test_pred.shape == test_true.shape

roc_auc = roc_auc_score(test_true, test_pred)
print(roc_auc)