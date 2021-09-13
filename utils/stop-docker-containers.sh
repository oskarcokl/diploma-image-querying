#!/bin/sh

echo "Stoping docker containers..."
docker container stop dev-postgres
docker container stop celery-rabbit
echo "Docker containers stop."
