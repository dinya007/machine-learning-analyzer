import json
import os
import time

import stomp

from jobs import Jobs

user = os.getenv("ACTIVEMQ_USER") or "admin"
password = os.getenv("ACTIVEMQ_PASSWORD") or "password"
host = os.getenv("ACTIVEMQ_HOST") or "localhost"
port = os.getenv("ACTIVEMQ_PORT") or 61613
id = 'machine-learning-analyzer'


class DatasetRequestListener(object):

    def __init__(self, destination, jobs):
        self.jobs = jobs
        self.connection = stomp.Connection(host_and_ports=[(host, port)])
        self.connection.set_listener('', self)
        self.connection.start()
        self.connection.connect(login=user, passcode=password)
        self.connection.subscribe(id=id, destination=destination, ack='auto')

    def on_error(self, headers, message):
        print('received an error %s' % message)

    def on_message(self, headers, message):
        print("Received %s message" % message)
        job_id = headers['correlation-id']
        message = json.loads(message)
        result = self.jobs.analyze(message['dataPath'], message['analyzePath'])
        result = json.dumps(result)
        self.connection.send(destination=headers['reply-to'], body=result,
                             headers={'amq-msg-type': 'text', 'correlation-id': job_id}, persistent='false')


class RegressionModelRequestListener(object):

    def __init__(self, destination, jobs):
        self.jobs = jobs
        self.connection = stomp.Connection(host_and_ports=[(host, port)])
        self.connection.set_listener('', self)
        self.connection.start()
        self.connection.connect(login=user, passcode=password)
        self.connection.subscribe(id=id, destination=destination, ack='auto')

    def on_error(self, headers, message):
        print('received an error %s' % message)

    def on_message(self, headers, message):
        print("Received %s message" % message)
        job_id = headers['correlation-id']
        message = json.loads(message)
        result = self.jobs.train(message['dataPath'], message['modelPath'])
        result = json.dumps(result)
        self.connection.send(destination=headers['reply-to'], body=result,
                             headers={'amq-msg-type': 'text', 'correlation-id': job_id}, persistent='false')


class RegressionPredictionRequestListener(object):

    def __init__(self, destination, jobs):
        self.jobs = jobs
        self.connection = stomp.Connection(host_and_ports=[(host, port)])
        self.connection.set_listener('', self)
        self.connection.start()
        self.connection.connect(login=user, passcode=password)
        self.connection.subscribe(id=id, destination=destination, ack='auto')

    def on_error(self, headers, message):
        print('received an error %s' % message)

    def on_message(self, headers, message):
        print("Received %s message" % message)
        predicting_job_id = headers['correlation-id']
        message = json.loads(message)
        result = self.jobs.predict(message['dataPath'], message['modelPath'], message['resultPath'])
        result = json.dumps(result)
        self.connection.send(destination=headers['reply-to'], body=result,
                             headers={'amq-msg-type': 'text', 'correlation-id': predicting_job_id}, persistent='false')


jobs = Jobs()

DatasetRequestListener('dataset-analyze-request', jobs)
RegressionModelRequestListener('regression-train-request', jobs)
RegressionPredictionRequestListener('regression-predict-request', jobs)

print("Waiting for messages...")
while 1:
    time.sleep(10)
