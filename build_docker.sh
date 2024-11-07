#!/bin/bash

echo "Building Docker image"
docker build -t greeter:latest .

echo "Creating Docker volumes"
docker volume create greeter_known_faces
docker volume create greeter_unknown_faces

echo "Stopping and removing existing container"
docker stop greeter
docker rm greeter

echo "Running Docker container"
docker run -d \
  --gpus all \
  --env-file ./.env \
  -p 8000:8000 \
  --volume greeter_known_faces:/app/known_faces \
  --volume greeter_unknown_faces:/app/unknown_faces \
  --restart unless-stopped --name greeter greeter