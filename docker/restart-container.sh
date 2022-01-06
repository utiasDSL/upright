#!/bin/sh
# Restart an existing container that has been stopped.

# Check that file containing the image name exists.
if ! [ -f image_name.txt ]; then
  echo "Could not find image_name.txt."
  exit 1
fi

# Get the name of the container image.
image=$(cat image_name.txt)

# Find the latest container.
container=$(docker ps -a --filter ancestor=$image --latest --quiet)

docker restart "$container"
