#!/bin/bash

GPU=0 image="garage:original" ./nvidia-docker run -it --rm -v `pwd`:/garage -w /garage:original  garage:original /bin/bash
