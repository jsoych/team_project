import os
from dotenv import load_dotenv
from zipfile import ZipFile


load_dotenv()
data_dir = os.getenv("RAW_DATA_DIR")

with ZipFile(os.path.join(data_dir,"chest-xray-pneumonia.zip")) as zipfile:
    zipfile.extractall(data_dir)