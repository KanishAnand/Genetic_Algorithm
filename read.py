import json
import requests
import numpy as np
import sys

with open('population.json') as f:
    data = json.loads(f.read())
    print(len(data))
# count = 0

# for line in Lines:
#     print(line.strip())
#     print("Line{}: {}".format(count, line.strip()))
