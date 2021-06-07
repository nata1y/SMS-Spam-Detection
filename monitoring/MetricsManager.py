"""
Only [a-zA-Z0-9:_] are valid in metric names, any other characters should be sanitized to an underscore.
The _sum, _count, _bucket and _total suffixes are used by Summaries, Histograms and Counters. Unless you’re producing one of those, avoid these suffixes.
_total is a convention for counters, you should use it if you’re using the COUNTER type.
The process_ and scrape_ prefixes are reserved.
Avoid type as a label name, it’s too generic and often meaningless. You should also try where possible to avoid names that are likely to clash with target labels, such as region, zone, cluster, availability_zone, az, datacenter, dc, owner, customer, stage, service, environment and env. If, however, that’s what the application calls some resource, it’s best not to cause confusion by renaming it.

"""
class Metric:
    namespace: str = None
    name: str = None
    value = None
    description: str = None

    def __init__(self, namespace: str, name: str, description: str, value):
        """
        Constructor of a metric
        :param namespace: Namespace of a metric (http, node, process for example)
        :param name: Name of the metric itself
        :param value: Initial value of the metric (empty string by default)
        :param description: Description of the metric (empty string by default)
        """
        self.namespace = namespace
        self.name = name
        self.value = value
        self.description = description

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.getFullName() == other.getFullName()
        return False

    def __str__(self):
        return "Metric[{}, {}, {}, {}]".format(self.getNamespace(),
                                               self.getName(),
                                               self.getDescription(),
                                               self.getValue()
                                               )

    def getFullName(self) -> str:
        """
        Get the full name of the metric: {namespace}_{name}
        https://prometheus.io/docs/practices/naming/
        Examples:
        - prometheus_notifications_total (specific to the Prometheus server)
        - process_cpu_seconds_total (exported by many client libraries)
        - http_request_duration_seconds (for all HTTP requests)
        :return: full name of the metric as displayed in prometheus
        """
        return "{}_{}".format(self.namespace, self.name)

    def getNamespace(self) -> str:
        return self.namespace

    def setNamespace(self, namespace: str):
        self.namespace = namespace

    def getName(self) -> str:
        return self.name

    def setName(self, name: str):
        self.name = name

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

    def getDescription(self):
        return self.description

    def setDescription(self, description: str):
        """
        Set the description.
        :param description: description of a metric.
        :return: void
        """
        self.description = description

    def getPrometheusString(self) -> str:
        """
        Get a string representing the metric that can be output to Prometheus .
        :return: A string summarising the metric that can be read by Prometheus
        """
        metricString: str = ""
        if self.description != "":
            metricString += "#{}\n".format(self.description)
        metricString += "{} {}\n".format(self.getFullName(), self.value)
        return metricString


class MetricsManager:
    # key= (fullname)
    # value = Metric
    metrics = None

    def __init__(self):
        self.metrics = {}

    def __getMetricKey(self, namespace: str, name: str) -> str:
        """
        Get the key of the corresponding metric.
        This is the same as the metric-naming for prometheus
        :return: unique key for the metric
        """
        return "{}_{}".format(namespace, name)

    def newMetric(self, namespace: str, name: str, description="", value=""):
        """
        Define a new metric for Prometheus
        :param namespace: Name space of the metric
        :param name: Name of the metric
        :param description: Description of the metric
        :param value: Initial value of the metric
        :return: void
        """
        key = self.__getMetricKey(namespace, name)
        metric: Metric = Metric(namespace, name, description, value)
        self.metrics.update({key: metric})

    def updateMetric(self, namespace: str, name: str, value):
        """
        Update an existing metric.
        If the metric does not exist, it creates a new one.
        :param namespace: Namespace of the metric
        :param name: Name of the metric
        :param value: New value of the metric
        :return: void
        """
        key = self.__getMetricKey(namespace, name)
        metric = self.metrics.get(key)
        if metric is None:
            metric = Metric(namespace, name, "", value)
        else:
            metric.setValue(value)
        self.metrics.update({key: metric})

    def getMetric(self, namespace: str, name: str) -> Metric:
        """
        Get a metric of choice. It returns None if no metric with corresponding namespace and name exists
        :param namespace: Namespace of the metric
        :param name: Name of the metric
        :return: Corresponding Metric
        """
        key = self.__getMetricKey(namespace, name)
        return self.metrics.get(key)

    def getPrometheusMetricsString(self) -> str:
        """
        Get a string sumarising all metrics that can be read by Prometheus
        :return: A string that can be read by Prometheus.
        """
        metrics: str = ""
        for (_, metric) in self.metrics.items():
            metrics += metric.getPrometheusString()
            print(metrics)
        return metrics