#!/bin/bash

import  sys
import pdb

class Sum:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sum(self):
        return self.a + self.b

sum_func = Sum(5,7)
print(sum_func.sum())
