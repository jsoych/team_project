import csv
import tensorflow as tf


batch_size = 32
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/raw/chest_xray/train",
    batch_size=batch_size
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/raw/chest_xray/test",
    batch_size=batch_size
)

vgg19 = tf.keras.applications.VGG19(
    include_top=False,
    pooling="avg"
)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

with open("data/processed/vgg19_train_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for image_batch, label_batch in iter(train_dataset):
        image_batch = preprocess_input(image_batch)
        feature_batch = vgg19(image_batch)
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])

with open("data/processed/vgg19_test_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for image_batch, label_batch in iter(test_dataset):
        image_batch = preprocess_input(image_batch)
        feature_batch = vgg19(image_batch)
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])