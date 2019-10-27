import json
import os
import sys

import stomp

user = os.getenv("ACTIVEMQ_USER") or "admin"
password = os.getenv("ACTIVEMQ_PASSWORD") or "password"
host = os.getenv("ACTIVEMQ_HOST") or "localhost"
port = os.getenv("ACTIVEMQ_PORT") or 61613
destination = sys.argv[1:2] or ["machine-learning-request"]
destination = destination[0]

messages = 10000
data = {"message": "Hello World from Python"}

conn = stomp.Connection(host_and_ports=[(host, port)])
conn.start()
conn.connect(login=user, passcode=password)
data = json.dumps(data)
conn.send(destination=destination, body=data, headers= {'amq-msg-type': 'text'},persistent='false')
# conn.send(destination=destination, headers={'transformation': 'jms-map-json'}, body=data, persistent='false')

# for i in range(0, 1000):
#     conn.send(body=data, destination=destination, persistent='false')

# conn.send("SHUTDOWN", destination=destination, persistent='false')

conn.disconnect()
