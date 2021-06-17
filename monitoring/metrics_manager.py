"""
Only [a-zA-Z0-9:_] are valid in metric names, any other characters should be sanitized to an
underscore. The _sum, _count, _bucket and _total suffixes are used by Summaries, Histograms
and Counters. Unless you’re producing one of those, avoid these suffixes. _total is a convention
for counters, you should use it if you’re using the COUNTER type. The process_ and scrape_ prefixes
are reserved. Avoid type as a label name, it’s too generic and often meaningless. You should also
try where possible to avoid names that are likely to clash with target labels, such as region,
zone, cluster, availability_zone, az, datacenter, dc, owner, customer, stage, service, environment
and env. If, however, that’s what the application calls some resource, it’s best not to cause
confusion by renaming it.
"""
class Metric:
    '''Class Metric for datastructure for Prometheus and Grafana.'''
    name: str = None
    value = None
    smoothed_value = None
    description: str = None

    def __init__(self, name: str, description: str, value, smoothed_value):
        """
        Constructor of a metric
        :param name: Name of the metric itself
        :param value: Initial value of the metric (empty string by default)
        :param description: Description of the metric (empty string by default)
        """
        self.name = name
        self.value = value
        self.smoothed_value = smoothed_value
        self.description = description

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.get_full_name() == other.get_full_name()
        return False

    def __str__(self):
        return "Metric[{}, {}, {}]".format(self.get_name(),
            self.get_description(), self.get_value())

    def get_full_name(self) -> str:
        """
        Get the full name of the metric: {name}
        https://prometheus.io/docs/practices/naming/
        Examples:
        - prometheus_notifications_total (specific to the Prometheus server)
        - process_cpu_seconds_total (exported by many client libraries)
        - http_request_duration_seconds (for all HTTP requests)
        :return: full name of the metric as displayed in prometheus
        """
        return "{}".format(self.name)

    def get_name(self) -> str:
        '''
        Get the name of the metric.
        :return: name string of metric.
        '''
        return self.name

    def set_name(self, name: str):
        '''
        Set the name of the metric.
        :param: name string of the metric.
        :return: void
        '''
        self.name = name

    def get_value(self, smoothed=True):
        '''
        Get the value of the metric.
        :param: smoothed boolean which value to retrieve.
        :return: value of smoothed value.
        '''
        if smoothed:
            return self.smoothed_value
        return self.value

    def set_value(self, value):
        '''
        Set the value of the metric.
        :param: value to set as metric
        :return: void
        '''
        self.value = value

    def set_smooth_value(self, value):
        '''
        Set smoothed value of the metric.
        :param: value to set as metric
        :return: void
        '''
        self.smoothed_value = value

    def get_description(self):
        '''
        Get the description of the metric.
        :return: A string with description
        '''
        return self.description

    def set_description(self, description: str):
        """
        Set the description.
        :param description: description of a metric.
        :return: void
        """
        self.description = description

    def get_prometheus_string(self) -> str:
        """
        Get a string representing the metric that can be output to Prometheus.
        :return: A string summarising the metric that can be read by Prometheus.
        """
        metric_string: str = ""
        if self.description != "":
            metric_string += "#{}\n".format(self.description)
        metric_string += "{} {}\n".format(self.get_full_name(), self.value)
        return metric_string


class MetricsManager:
    '''Manager class for the metrics, listing them.'''
    metrics: dict()

    def __init__(self):
        self.metrics = {}

    @staticmethod
    def _get_metric_key(name: str) -> str:
        """
        Get the key of the corresponding metric.
        This is the same as the metric-naming for prometheus
        :return: unique key for the metric
        """
        return "{}".format(name)

    def new_metric(self, name: str, description="", value="", smooth_value=""):
        """
        Define a new metric for Prometheus
        :param name: Name of the metric
        :param description: Description of the metric
        :param value: Initial value of the metric
        :return: void
        """
        key = MetricsManager._get_metric_key(name)
        metric: Metric = Metric(name, description, value, smooth_value)
        self.metrics.update({key: metric})

    def update_metric(self, name: str, value, smooth_value=""):
        """
        Update an existing metric.
        If the metric does not exist, it creates a new one.
        :param name: Name of the metric
        :param value: New value of the metric
        :return: void
        """
        key = MetricsManager._get_metric_key(name)
        metric = self.metrics.get(key)
        if metric is None:
            metric = Metric(name, "", value, smooth_value)
        else:
            metric.setValue(value)
            metric.setSmoothValue(smooth_value)
        self.metrics.update({key: metric})

    def get_metric(self, name: str) -> Metric:
        """
        Get a metric of choice. It returns None if no metric with corresponding name exists
        :param name: Name of the metric
        :return: Corresponding Metric
        """
        key = MetricsManager._get_metric_key(name)
        return self.metrics.get(key)

    def get_prometheus_metrics_string(self) -> str:
        """
        Get a string sumarising all metrics that can be read by Prometheus
        :return: A string that can be read by Prometheus.
        """
        metrics: str = ""
        for (_, metric) in self.metrics.items():
            metrics += metric.getPrometheusString()
        return metrics
