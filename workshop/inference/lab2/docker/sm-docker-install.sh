#!/bin/bash

sudo apt-get update
sudo apt-get install ca-certificates curl gnupg -y
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

sudo echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

VERSION_STRING=5:20.10.24~3-0~ubuntu-${VERSION_CODENAME:-jammy}
sudo apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]

if [ -z "${DOCKER_HOST}" ]; then
  export DOCKER_HOST="unix:///docker/proxy.sock"
fi

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]
docker version