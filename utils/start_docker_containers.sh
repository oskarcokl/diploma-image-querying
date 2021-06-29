#!/bin/sh

echo "Starting docker containers..."
docker container start jolly_shirley
docker container start rabbitmq
echo "Docker containers started."