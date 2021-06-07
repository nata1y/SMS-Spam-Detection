from random import randint

from flask import Flask
from flasgger import Swagger

from monitoring.MetricsManager import MetricsManager

app = Flask(__name__)
swagger = Swagger(app)

metricsManager: MetricsManager = MetricsManager()

# http://localhost:8080/metrics/update/namespace/name/newvalue
@app.route('/metrics/update/<namespace>/<name>/<value>')
def updateMetric(namespace: str, name: str, value):
    old: str = str(metricsManager.getMetric(namespace, name))
    metricsManager.updateMetric(namespace, name, value)
    new: str = str(metricsManager.getMetric(namespace, name))
    return "Updated metric from {} to {}".format(old, new)

# http://localhost:8080/metrics/randomupdate/namespace/name
@app.route('/metrics/randomupdate/<namespace>/<name>')
def randomUpdateMetric(namespace: str, name: str):
    old: str = str(metricsManager.getMetric(namespace, name))
    value = randint(0, 100)
    metricsManager.updateMetric(namespace, name, value)
    new: str = str(metricsManager.getMetric(namespace, name))
    return "Randomly updated metric from {} to {}".format(old, new)


# http://localhost:8080/metrics/new/namespace/name/description/value
@app.route('/metrics/new/<namespace>/<name>/<description>/<value>')
def newMetric(namespace: str, name: str, description: str, value):
    metricsManager.newMetric(namespace, name, description, value)
    return "Successfully added a new metric: {}".format(metricsManager.getMetric(namespace, name))

# http://localhost:8080/metrics/get/namespace/name
@app.route('/metrics/get/<namespace>/<name>')
def getMetric(namespace: str, name: str):
    return "Got metric: {}".format(metricsManager.getMetric(namespace, name))

# http://localhost:8080/metrics
@app.route('/metrics', methods=['GET'])
def metrics_dump():
    return metricsManager.getPrometheusMetricsString()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
