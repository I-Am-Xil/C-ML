#!/usr/bin/bash

set -xe
clang nn.c -o bin/nn -lm -Wall -Wextra
./bin/nn
