#!/usr/bin/bash

set -xe
clang gtwice.c -o bin/gtwice -lm -Wall -Wextra
./bin/gtwice
