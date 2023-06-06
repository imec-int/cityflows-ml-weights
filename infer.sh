#!/bin/bash

for suffix in {0..23}; do echo coordinate file part $suffix; /opt/anaconda3/envs/straatvinken-minimal/bin/python src/aicityflowsstraatvinken/03-infer.py --suffix $suffix; done
