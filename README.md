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
$ docker run -it --rm -v "$(pwd)":/root/project -p "8080:8080" docker-sms
```

c) Run various scripts

```
$ python src/get_data.py
$ python src/read_data.py
$ python src/text_preprocessing.py
$ python src/text_classification.py
```

d) Serve the model as a REST API

NOTE: add `host="0.0.0.0"` to `app.run` in `src/serve_model.py`. (default 127.0.0.1 does not work in Docker)

```
$ python src/serve_model.py
```

You can test the API using the following:

```
curl -X POST "http://127.0.0.1:8080/predict" -H  "accept: application/json" -d "{sms: hello world!}"
```

Alternatively, you can access the UI using your browser: http://127.0.0.1:8080/apidocs

[*Release Engineering for Machine Learning Applications* (REMLA)]: https://se.ewi.tudelft.nl/remla/ 
[Prof. Luís Cruz]: https://luiscruz.github.io/
[Prof. Sebastian Proksch]: https://proks.ch/
