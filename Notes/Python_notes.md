## Python Important Notes


### Ternary Operator

```py
# Program to demonstrate conditional operator 
a, b = 10, 20
  
# Copy value of a in min if a < b else copy b 
min = a if a < b else b 
  
print(min) 
# Output: 10
```

### Convert String to list

```py
st = "SimpleString"

s_list = list(st)
#Output: ['S', 'i', 'm', 'p', 'l', 'e', 'S', 't', 'r', 'i', 'n', 'g']
```

#### Dictionary Comprehension

```py
# create a dictionary

x = {str(i): i for i in range(5)}

#Output: {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1}
```