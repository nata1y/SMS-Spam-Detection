# SMS Spam Detection Using Machine Learning

This project is used a starting point for the course [*Release Engineering for Machine Learning Applications* (REMLA)] taught at the Delft University of Technology by [Prof. Luís Cruz] and [Prof. Sebastian Proksch].

The codebase was originally adapted from: https://github.com/rohan8594/SMS-Spam-Detection

## Instructions for Compiling

a) Clone repo.

```
$ git clone https://github.com/luiscruz/SMS-Spam-Detection.git
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

e) Run the regression model

Retrieves and splits the dataset from the first 1000 labels on which the model is trained. 
Get the predictions via HTTP requests from the model like in an actual deployment setup.
Run LogisticRegression on the predicted set, actual set and model set for comparison.

NOTE: to get predictions from inside another docker container use `docker run -it --rm -v "$(pwd)":/root/project --net=host docker-sms`, since the port is already opened for the server, but you want to connect to its local network.

```
$ python regression_model/get_data.py
$ python regression_model/read_data.py
$ python regression_model/get_predictions.py
$ python regression_model/predict_data.py
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

Run Prometheus using docker:
```
docker run --rm -p 9090:9090 -v ${PWD}/prometheus_data/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```