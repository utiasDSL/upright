#!/usr/bin/env bash

# Check that file containing the image name exists.
if ! [ -f image_name.txt ]; then
  echo "Could not find image_name.txt."
  exit 1
fi

# exit immediately if any commands fail
set -e

# Run the container with shared X11 (i.e. we can run GUI programs from within
# the container).
#
# --privileged and -v /dev/bus/usb:/dev/bus/usb are meant to give access to USB
# devices. See
# https://stackoverflow.com/questions/24225647/docker-any-way-to-give-access-to-host-usb-or-serial-device
#
# Note: /dev/bus/usb doesn't actually exist on macOS, if you're bold enough to
# try this on one. I'm not sure what the alternative is (or if one is needed).
# Good luck.
docker run\
  --net=host\
  -e SHELL\
  -e DISPLAY\
  -e DOCKER=1\
  -e TERM\
  --privileged\
  --device /dev/dri\
  --ulimit rtprio=99\
  --cap-add=sys_nice\
  -v "/dev/bus/usb:/dev/bus/usb"\
  -v "$HOME:$HOME:rw"\
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"\
  -it $(cat image_name.txt) $SHELL
