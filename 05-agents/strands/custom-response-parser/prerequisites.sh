#!/bin/bash

sudo apt-get install zip -y

git clone https://github.com/awslabs/ml-container-creator
cd ml-container-creator

npm install && npm link

yo --generators