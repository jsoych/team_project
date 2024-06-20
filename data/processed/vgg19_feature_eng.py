import csv
import tensorflow as tf


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/raw/chest_xray/train"
)

vgg19 = tf.keras.applications.VGG19(
    include_top=False,
    pooling="avg"
)

csvfile = open("data/processed/vgg19_train_data.csv", "w", newline="")
writer = csv.writer(csvfile)
for image_batch, label_batch in iter(train_dataset):
    feature_batch = vgg19(image_batch)
    for i in range(32):
        writer.writerow([*list(feature_batch[i].numpy()),label_batch[i].numpy()])
csvfile.close()