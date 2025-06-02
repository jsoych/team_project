import os
import csv
import tensorflow as tf

RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')

batch_size = 32
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(RAW_DATA_DIR,"chest_xray/train"),
    batch_size=batch_size,
    shuffle=True,
    seed=0
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(RAW_DATA_DIR,"chest_xray/test"),
    batch_size=batch_size,
    shuffle=True,
    seed=0
)

# Load the pretrained model, and the corresponding preprocessing function
model = tf.keras.applications.VGG16(
    include_top=False,
    pooling='avg'
)

preprocess_input = tf.keras.applications.vgg16.preprocess_input

# Engineer features for each datum in train and test datasets, then write
# the features with the label to a csv file
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR')

if not(os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'vgg16'))):
    os.mkdir(os.path.join(PROCESSED_DATA_DIR, 'vgg16'))

with open(os.path.join(PROCESSED_DATA_DIR, 'vgg16/train_data.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    col_names = ['x{}'.format(i) for i in range(model.output.shape[-1])]
    col_names.append("y")
    writer.writerow(col_names)
    for image_batch, label_batch in iter(train_dataset):
        # Engineer features
        image_batch = preprocess_input(image_batch)
        feature_batch = model(image_batch)

        # Write them to the file
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])

with open(os.path.join(PROCESSED_DATA_DIR, 'vgg16/test_data.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    col_names = ["x{}".format(i) for i in range(model.output.shape[-1])]
    col_names.append("y")
    writer.writerow(col_names)
    for image_batch, label_batch in iter(test_dataset):
        # Engineer features
        image_batch = preprocess_input(image_batch)
        feature_batch = model(image_batch)

        # Write them to the file
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])
