import os
import csv
import tensorflow as tf
from dotenv import load_dotenv


load_dotenv()
raw_data_dir = os.getenv("RAW_DATA_DIR")

batch_size = 32
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(raw_data_dir,"chest_xray/train"),
    batch_size=batch_size,
    shuffle=True,
    seed=0
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(raw_data_dir,"chest_xray/test"),
    batch_size=batch_size,
    shuffle=True,
    seed=0
)

# Load the pretrained model, and the corresponding preprocessing function
model = tf.keras.applications.ResNet50(
    include_top=False,
    pooling="avg"
)

preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Engineer features for each datum in train, and test dataset, then write
# the features with the label to a csv file
data_dir = os.getenv("PROCESSED_DATA_DIR")

with open(os.path.join(data_dir,"data_3/train_data.csv"), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    col_names = ["x{}".format(i) for i in range(model.output.shape[-1])]
    col_names.append("y")
    writer.writerow(col_names)
    for image_batch, label_batch in iter(train_dataset):
        image_batch = preprocess_input(image_batch)
        feature_batch = model(image_batch)
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])

with open(os.path.join(data_dir,"data_3/test_data.csv"), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    col_names = ["x{}".format(i) for i in range(model.output.shape[-1])]
    col_names.append("y")
    writer.writerow(col_names)
    for image_batch, label_batch in iter(test_dataset):
        image_batch = preprocess_input(image_batch)
        feature_batch = model(image_batch)
        for i in range(min(batch_size,image_batch.shape[0])):
            writer.writerow([*feature_batch[i].numpy(),label_batch[i].numpy()])