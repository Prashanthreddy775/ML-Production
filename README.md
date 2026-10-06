\# Iris ML Prediction API



This project demonstrates a machine learning deployment workflow using the Iris dataset and a Logistic Regression model.



\## What the model predicts



The model predicts the species of an Iris flower using four features:



\* Sepal length

\* Sepal width

\* Petal length

\* Petal width



The possible predictions are Setosa, Versicolor, and Virginica.



\## API Endpoints



\### GET /health



Checks whether the API is running and whether the trained model was loaded.



Example response:



```json

{

&#x20; "status": "ok",

&#x20; "model\_loaded": true

}

```



\### POST /predict



Accepts Iris flower measurements and returns the predicted class.



Example request:



```json

{

&#x20; "sepal\_length": 5.1,

&#x20; "sepal\_width": 3.5,

&#x20; "petal\_length": 1.4,

&#x20; "petal\_width": 0.2

}

```



Example response:



```json

{

&#x20; "prediction": 0,

&#x20; "class": "setosa"

}

```



\## How to run locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Start the API:



```bash

uvicorn main:app --reload

```



Open the API documentation:



http://127.0.0.1:8000/docs



\## Model



The trained machine learning model is saved as `model.pkl`.



