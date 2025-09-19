import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://localhost:30885')

model_name = 'pneumonia_classifier'
model_version = 'None'

model_uri = f'models:/{model_name}/{model_version}'

model = mlflow.sklearn.load_model(model_uri)

print(model.get_params())