#!/bin/sh

echo "Starting docker containers..."
docker container start dev-postgres
docker container start celery-rabbit
echo "Docker containers started."
