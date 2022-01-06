#!/usr/bin/env bash

# Do not use sudo because this will give the docker user the wrong privileges.
if [ -n "$SUDO_USER" ]; then
  echo "Do not run with sudo."
  exit 1
fi

# Check that file containing the image name exists.
if ! [ -f image_name.txt ]; then
  echo "Could not find image_name.txt."
  exit 1
fi

# Build the docker image
docker build\
  --build-arg user=$USER\
  --build-arg pass="foo"\
  --build-arg uid=$UID\
  --build-arg home=$HOME\
  --build-arg shell=$SHELL\
  -t $(< image_name.txt) .
