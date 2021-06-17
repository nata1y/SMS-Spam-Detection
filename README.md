![pylint](https://nata1y.gitlab.io/SMS-Spam-Detection/pyling.svg)

# SMS Spam Detection Using Machine Learning

This project is used a starting point for the course [*Release Engineering for Machine Learning Applications* (REMLA)] taught at the Delft University of Technology by [Prof. Luís Cruz] and [Prof. Sebastian Proksch].

The codebase was originally adapted from: https://github.com/rohan8594/SMS-Spam-Detection

## Instructions for Compiling

a) Clone repo.

```
$ git clone https://gitlab.com/nata1y/SMS-Spam-Detection
$ cd SMS-Spam-Detection
$ mkdir output
```

b) Install dependencies.

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

b2) Alternatively, use Docker for dependencies and volumes.

```
$ docker build --progress plain . -t docker-sms
$ docker run -it --rm -v ${PWD}:/root/project -p "8080:8080" docker-sms
~# $ cd project
```

c) Run various scripts

```
$ python train_model/get_data.py
$ python train_model/read_data.py
$ python train_model/text_preprocessing.py
$ python train_model/text_classification.py
```

d) Serve the model as a REST API

NOTE: add `host="0.0.0.0"` parameter to `app.run` call in `deploy_model/serve_model.py`. (default 127.0.0.1 does not work in Docker)

```
$ python deploy_model/serve_model.py
```

b3) Or, use docker-compose and automatically train and host.

```
$ docker-compose -f docker-compose.train.yml build
$ docker-compose -f docker-compose.train.yml up -d && ./get_training_data.sh && docker-compose -f docker-compose.train.yml down
```

Or, if you use windows run the commands inside the script instead of the script.

Use the following command if you want to run the system without retraining everything
```
docker-compose up --build
```

e) Production endpoint

Retrieves and splits the dataset from the first 1000 labels on which the model is trained. 
Get the predictions via HTTP requests from the model like in an actual deployment setup.

NOTE: to get predictions from inside another docker container use `docker run -it --rm -v "$(pwd)":/root/project --net=host docker-sms`, since the port is already opened for the server, but you want to connect to its local network.
OR: if you use docker-compose run `docker exec -it <container_id> bash` to run the deploy script.

```
$ python production_endpoint/get_data.py
$ python production_endpoint/generate_drifts.py
$ python production_endpoint/get_predictions.py
```

You can test the API using the following:

```
$ curl -X POST "http://127.0.0.1:8080/predict" -H  "accept: application/json" -d "{sms: hello world!}"
or
$ curl -X POST "http://127.0.0.1:8080/predict" -H  "Content-Type: application/json" -d '{"sms": "hello world!"}'
```

Alternatively, you can access the UI using your browser: http://127.0.0.1:8080/apidocs

[*Release Engineering for Machine Learning Applications* (REMLA)]: https://se.ewi.tudelft.nl/remla/ 
[Prof. Luís Cruz]: https://luiscruz.github.io/
[Prof. Sebastian Proksch]: https://proks.ch/
