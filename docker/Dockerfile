FROM osrf/ros:noetic-desktop-full

# Arguments
ARG user
ARG pass
ARG uid
ARG home
ARG shell

# Basic Utilities
RUN apt-get update && apt-get install -y git zsh curl screen tree sudo ssh synaptic vim apt-utils ripgrep

# Python.
RUN apt-get install -y python-dev python3-dev python3-pip
RUN pip3 install --upgrade pip

# Additional development tools
RUN apt-get install -y x11-apps build-essential

# OCS2 dependencies
RUN apt-get install -y libeigen3-dev libglpk-dev python3-catkin-tools python3-osrf-pycommon ros-noetic-pybind11-catkin python3-tk

# needed for some of the OCS2 examples
RUN apt-get install -y expect

# TODO: needs to be done manually for now
#   ideally, I'd remove the gnome-terminal dependency completely (it came with
#   the OCS2 mobile manipulator example)
# RUN apt-get install -y gnome-terminal

# Make SSH available
EXPOSE 22

# Mount the user's home directory
VOLUME "${home}"

# Clone user into docker image and set up X11 sharing
# TODO for some reason can't get sudo access on the non-root user
RUN \
  echo "${user}:x:${uid}:${uid}:${user},,,:${home}:${shell}" >> /etc/passwd && \
  echo "${user}:x:${uid}:" >> /etc/group && \
  echo "${user} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${user}" && \
  chmod 0440 "/etc/sudoers.d/${user}"

# this is seemingly needed to allow the user to use sudo
RUN echo "${user}:${pass}" | chpasswd

# Switch to user
USER "${user}"

# This is required for sharing Xauthority
ENV QT_X11_NO_MITSHM=1

WORKDIR ${home}
