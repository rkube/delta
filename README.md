aDaptive rEaL Time Analysis of big fusion data

This project implements a client-server model for analysis of streamed data from
fusion experiments or large-scale simulations.

Implemented as part of "Adaptive near-real time net-worked analysis of big
fusion data", (FY18).


Start/Stop zookeper and kafka: https://gist.github.com/piatra/0d6f7ad1435fa7aa790a
#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Please supply topic name"
  exit 1
fi

nohup bin/zookeeper-server-start.sh -daemon config/zookeeper.properties > /dev/null 2>&1 &
sleep 2
nohup bin/kafka-server-start.sh -daemon config/server.properties > /dev/null 2>&1 &
sleep 2

bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic $1
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic parsed
