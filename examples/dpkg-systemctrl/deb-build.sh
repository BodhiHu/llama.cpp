#!/bin/bash

set -ex

VERSION=0.0.7

dpkg-deb --build -Znone -Snone llama-server
mv llama-server.deb llama-server-${VERSION}-$(uname -m).deb
