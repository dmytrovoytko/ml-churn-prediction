FROM python:3.11.9-slim

# RUN apt clean
# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
# RUN apt-get -y install curl
# RUN apt-get -y install unzip nano

RUN pip install --no-cache-dir -U pip

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./model ./model 

COPY app.py . 
COPY predict.py . 
COPY preprocess.py . 
COPY settings.py . 
COPY train_model.py . 
COPY test-api.py . 

EXPOSE 5555

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5555", "app:app" ]
