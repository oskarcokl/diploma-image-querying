#!/bin/sh

celery -A cd_tree_celery worker --loglevel=info -n cd_tree_worker &
celery -A cnn_celery worker --loglevel=info -n cnn_celery &
