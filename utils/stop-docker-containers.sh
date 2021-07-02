#!/bin/sh

echo "Stoping docker containers..."
docker container stop jolly_shirley
docker container stop rabbitmq
echo "Docker containers stop."