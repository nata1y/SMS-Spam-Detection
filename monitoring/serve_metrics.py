'''Metrics routes for Prometheus to retrieve values from.'''
from random import randint

from flask import Flask
from flasgger import Swagger

from monitoring.metrics_manager import MetricsManager

app = Flask(__name__)
swagger = Swagger(app)

metricsManager: MetricsManager = MetricsManager()

# http://localhost:8080/metrics/update/namespace/name/newvalue
@app.route('/metrics/update/<name>/<value>')
def update_metric(name: str, value):
    '''Update a metric.'''
    old: str = str(metricsManager.get_metric(name))
    metricsManager.update_metric(name, value)
    new: str = str(metricsManager.get_metric(name))
    return "Updated metric from {} to {}".format(old, new)

# http://localhost:8080/metrics/randomupdate/namespace/name
@app.route('/metrics/randomupdate/<name>')
def random_update_metric(name: str):
    '''Update metric with random value between 0 and 100.'''
    old: str = str(metricsManager.get_metric(name))
    value = randint(0, 100)
    metricsManager.update_metric(name, value)
    new: str = str(metricsManager.get_metric(name))
    return "Randomly updated metric from {} to {}".format(old, new)


# http://localhost:8080/metrics/new/namespace/name/description/value
@app.route('/metrics/new/<name>/<description>/<value>')
def new_metric(name: str, description: str, value):
    '''Creates a new metric.'''
    metricsManager.new_metric(name, description, value)
    return "Successfully added a new metric: {}".format(metricsManager.get_metric(name))

# http://localhost:8080/metrics/get/namespace/name
@app.route('/metrics/get/<name>')
def get_metric(name: str):
    '''Retrieves a metric.'''
    return "Got metric: {}".format(metricsManager.get_metric(name))

# http://localhost:8080/metrics
@app.route('/metrics', methods=['GET'])
def metrics_dump():
    '''Dump the metric to a string.'''
    return metricsManager.get_prometheus_metrics_string()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
