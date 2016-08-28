Python Tutorial
----
This is a simple tutorial to remember python functionalities and libraries functions


```python
# Start with Simple Hello world
print("Hello Himanshu!")
```

    Hello Himanshu!


### Creating Variable

Python doesn't have any specific way to declare variable.

Variable get created once you assign any value to them


```python
x = 7
y = "Ramesh"
print(x)
print(y)
```

    7
    Ramesh


### Variable Names

A variable can have a short name (like x and y) or a more descriptive name (age, carname, total_volume). Rules for Python variables:

* A variable name must start with a letter or the underscore character
* A variable name cannot start with a number
* A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
* Variable names are case-sensitive (age, Age and AGE are three different variables)


```python
# Legal variable names
myvar = "John"
my_var = "John"
_my_var = "John"
myVar = "John"
MYVAR = "John"
myvar2 = "John"

```


```python
#Illegal variable names
# 2myvar = "John"
# my-var = "John"
# my var = "John"
```

### The global Keyword

Normally, when you create a variable inside a function, that variable is local, and can only be used inside that function.

To create a global variable inside a function, you can use the global keyword.


```python
#If you use the global keyword, the variable belongs to the global scope:
def myfunc():
  global x
  x = "fantastic"

myfunc()

print("Python is " + x)
```

    Python is fantastic


### Python Data Types

<table class="w3-table">
  <tbody><tr>
    <td style="width:160px;">Text Type:</td>
    <td><code class="w3-codespan">str</code></td>
  </tr>
  <tr>
    <td>Numeric Types:</td>
    <td><code class="w3-codespan">int</code>, <code class="w3-codespan">float</code>,
    <code class="w3-codespan">complex</code></td>
  </tr>
  <tr>
    <td>Sequence Types:</td>
    <td><code class="w3-codespan">list</code>, <code class="w3-codespan">tuple</code>, 
    <code class="w3-codespan">range</code></td>
  </tr>
  <tr>
    <td>Mapping Type:</td>
    <td><code class="w3-codespan">dict</code></td>
  </tr>
  <tr>
    <td>Set Types:</td>
    <td><code class="w3-codespan">set</code>, <code class="w3-codespan">frozenset</code></td>
  </tr>
  <tr>
    <td>Boolean Type:</td>
    <td><code class="w3-codespan">bool</code></td>
  </tr>
  <tr>
    <td>Binary Types:</td>
    <td><code class="w3-codespan">bytes</code>, <code class="w3-codespan">bytearray</code>, 
    <code class="w3-codespan">memoryview</code></td>
  </tr>
</tbody></table>


```python
x = complex(1j)

#display x:
print(x)

#display the data type of x:
print(type(x)) 
```

    1j
    <class 'complex'>


### Python Numbers

There are three numeric types in Python:

* int
* float
* complex


```python
x = 1    # int
y = 2.8  # float
z = 1j   # complex
```


```python
x = 1.10
y = 1.0
z = -35.59

print(type(x))
print(type(y))
print(type(z))
```

    <class 'float'>
    <class 'float'>
    <class 'float'>


> Note: You cannot convert complex numbers into another number type.

### Strings

Strings in python are surrounded by either single quotation marks, or double quotation marks.

'hello' is the same as "hello".

You can display a string literal with the _print()_ function:


```python
a = "Hello"
print(a)
```

    Hello


### Multiline Strings
You can assign a multiline string to a variable by using three quotes:


```python
a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)
```

    Lorem ipsum dolor sit amet,
    consectetur adipiscing elit,
    sed do eiusmod tempor incididunt
    ut labore et dolore magna aliqua.


>Note: in the result, the line breaks are inserted at the same position as in the code.

### Strings are Arrays
Like many other popular programming languages, strings in Python are arrays of bytes representing unicode characters.

However, Python does not have a character data type, a single character is simply a string with a length of 1.

Square brackets can be used to access elements of the string.


```python
a = "Hello, World!"
print(a[1])
```

    e


### Looping through a string


```python
for x in "banana":
  print(x)
```

    b
    a
    n
    a
    n
    a



```python
a = "Hello Himanshu"
b = a.upper()
c = a.lower()
display(a,b, c)
```


    'Hello Himanshu'



    'HELLO HIMANSHU'



    'hello himanshu'


### String Methods

|Method|Description|
|--- |--- |
|capitalize()|Converts the first character to upper case|
|casefold()|Converts string into lower case|
|center()|Returns a centered string|
|count()|Returns the number of times a specified value occurs in a string|
|encode()|Returns an encoded version of the string|
|endswith()|Returns true if the string ends with the specified value|
|expandtabs()|Sets the tab size of the string|
|find()|Searches the string for a specified value and returns the position of where it was found|
|format()|Formats specified values in a string|
|format_map()|Formats specified values in a string|
|index()|Searches the string for a specified value and returns the position of where it was found|
|isalnum()|Returns True if all characters in the string are alphanumeric|
|isalpha()|Returns True if all characters in the string are in the alphabet|
|isdecimal()|Returns True if all characters in the string are decimals|
|isdigit()|Returns True if all characters in the string are digits|
|isidentifier()|Returns True if the string is an identifier|
|islower()|Returns True if all characters in the string are lower case|
|isnumeric()|Returns True if all characters in the string are numeric|
|isprintable()|Returns True if all characters in the string are printable|
|isspace()|Returns True if all characters in the string are whitespaces|
|istitle()|Returns True if the string follows the rules of a title|
|isupper()|Returns True if all characters in the string are upper case|
|join()|Joins the elements of an iterable to the end of the string|
|ljust()|Returns a left justified version of the string|
|lower()|Converts a string into lower case|
|lstrip()|Returns a left trim version of the string|
|maketrans()|Returns a translation table to be used in translations|
|partition()|Returns a tuple where the string is parted into three parts|
|replace()|Returns a string where a specified value is replaced with a specified value|
|rfind()|Searches the string for a specified value and returns the last position of where it was found|
|rindex()|Searches the string for a specified value and returns the last position of where it was found|
|rjust()|Returns a right justified version of the string|
|rpartition()|Returns a tuple where the string is parted into three parts|
|rsplit()|Splits the string at the specified separator, and returns a list|
|rstrip()|Returns a right trim version of the string|
|split()|Splits the string at the specified separator, and returns a list|
|splitlines()|Splits the string at line breaks and returns a list|
|startswith()|Returns true if the string starts with the specified value|
|strip()|Returns a trimmed version of the string|
|swapcase()|Swaps cases, lower case becomes upper case and vice versa|
|title()|Converts the first character of each word to upper case|
|translate()|Returns a translated string|
|upper()|Converts a string into upper case|
|zfill()|Fills the string with a specified number of 0 values at the beginning|

### Python Lists

Lists are used to store multiple items in a single variable.

Lists are one of 4 built-in data types in Python used to store collections of data, the other 3 are Tuple, Set, and Dictionary, all with different qualities and usage.

Lists are created using square brackets:




```python
# Creating List
thislist = ["apple", "banana", "cherry"]
print(thislist)
```

    ['apple', 'banana', 'cherry']


#### Changeable
The list is changeable, meaning that we can change, add, and remove items in a list after it has been created.

#### List Length

To determine how many items a list has, use the len() function:


```python
thislist = ["apple", "banana", "cherry"]
print(len(thislist))
```

    3


#### The _list()_ Constructor
It is also possible to use the list() constructor when creating a new list.


```python
thislist = list(("apple", "banana", "cherry")) # note the double round-brackets
print(thislist)
```

    ['apple', 'banana', 'cherry']


### Python Collections (Arrays)
There are four collection data types in the Python programming language:

1. **List** is a collection which is ordered and changeable. Allows duplicate members.
1. **Tuple** is a collection which is ordered and unchangeable. Allows duplicate members.
1. **Set** is a collection which is unordered and unindexed. No duplicate members.
1. **Dictionary** is a collection which is ordered* and changeable. No duplicate members.

>As of Python version 3.7, dictionaries are ordered. In Python 3.6 and earlier, dictionaries are unordered.

#### Access Items
List items are indexed and you can access them by referring to the index number:


```python
thislist = ["apple", "banana", "cherry"]
print(thislist[1])
```

    banana


#### Negative Indexing
Negative indexing means start from the end

-1 refers to the last item, -2 refers to the second last item etc.


```python
thislist = ["apple", "banana", "cherry"]
print(thislist[-1])
print(thislist[-2])
```

    cherry
    banana


#### Range of Indexes
You can specify a range of indexes by specifying where to start and where to end the range.

When specifying a range, the return value will be a new list with the specified items.


```python
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:5])
```

    ['cherry', 'orange', 'kiwi']


>Note: The search will start at index 2 (included) and end at index 5 (not included).

##### By leaving out the start value, the range will start at the first item:


```python
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[:4])
```

    ['apple', 'banana', 'cherry', 'orange']


##### By leaving out the end value, the range will go on to the end of the list:


```python
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:])
```

    ['cherry', 'orange', 'kiwi', 'melon', 'mango']


####Range of Negative Indexes
Specify negative indexes if you want to start the search from the end of the list:


```python
# This example returns the items from "orange" (-4) to, but NOT including "mango" (-1):
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[-4:-1])
```

    ['orange', 'kiwi', 'melon']


#### Check if Item Exists
To determine if a specified item is present in a list use the in keyword:


```python
#Check if "apple" is present in the list:
thislist = ["apple", "banana", "cherry"]
if "apple" in thislist:
  print("Yes, 'apple' is in the fruits list")
```

    Yes, 'apple' is in the fruits list


### Insert Items
To insert a new list item, without replacing any of the existing values, we can use the insert() method.

The insert() method inserts an item at the specified index:


```python
#Insert "watermelon" as the third item:
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
print(thislist)
```

    ['apple', 'banana', 'watermelon', 'cherry']


>Note: As a result of the example above, the list will now contain 4 items.

###Python - Add List Items

#### Append Items
To add an item to the end of the list, use the append() method:


```python
#Using the append() method to append an item:
thislist = ["apple", "banana", "cherry"]
thislist.append("orange")
print(thislist)
```

    ['apple', 'banana', 'cherry', 'orange']


### Extend List
To append elements from another list to the current list, use the extend() method.


```python
#Add the elements of tropical to thislist:
thislist = ["apple", "banana", "cherry"]
tropical = ["mango", "pineapple", "papaya"]
thislist.extend(tropical)
print(thislist)
```

    ['apple', 'banana', 'cherry', 'mango', 'pineapple', 'papaya']


####Add Any Iterable

The extend() method does not have to append lists, you can add any iterable object (tuples, sets, dictionaries etc.).


```python
thislist = ["apple", "banana", "cherry"]
thistuple = ("kiwi", "orange")
thislist.extend(thistuple)
print(thislist)
```

    ['apple', 'banana', 'cherry', 'kiwi', 'orange']


#### Remove Specified Item

The remove() method removes the specified item.


```python
#Remove "banana"
thislist = ["apple", "banana", "cherry"]
thislist.remove("banana")
print(thislist)
```

    ['apple', 'cherry']


#### Remove Specified Index

The pop() method removes the specified index.


```python
thislist = ["apple", "banana", "cherry"]
thislist.pop(2)
print(thislist)
```

    ['apple', 'banana']


The del keyword also removes the specified index


```python
thislist = ["apple", "banana", "cherry"]
del thislist[0]
print(thislist)
```

    ['banana', 'cherry']


The del keyword can also delete the list completely.


```python
thislist = ["apple", "banana", "cherry"]
del thislist
# print(thislist) this will give error as del cleared instance of thislist
```

#### Clear the List

The clear() method empties the list.

The list still remains, but it has no content.


```python
thislist = ["apple", "banana", "cherry"]
thislist.clear()
print(thislist)
```

    []


##### Loop Through a List

You can loop through the list items by using a for loop:


```python
thislist = ["apple", "banana", "cherry"]
for x in thislist:
  print(x)
```

    apple
    banana
    cherry


#### Looping Using List Comprehension
List Comprehension offers the shortest syntax for looping through lists:


```python
newlist = ["apple", "banana", "cherry"]
display([print(x) for x in newlist])
```

    apple
    banana
    cherry



    [None, None, None]


### Python - List Comprehension

List comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list.


```python
fruits = ["apple", "banana", "cherry", "kiwi", "mango"]

newlist = [x for x in fruits if "a" in x]

print(newlist)
```

    ['apple', 'banana', 'mango']


#### List Comprehension Syntax

```python
newlist = [expression for item in iterable if condition == True]
```

#### Condition in list comprehension

The condition is like a filter that only accepts the items that valuate to True.


```python
newlist = [x for x in fruits if x != "apple"]
```

### Python - Sort Lists
**Sort List Alphanumerically**

List objects have a sort() method that will sort the list alphanumerically, ascending, by default:


```python
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(thislist)
```

    ['banana', 'kiwi', 'mango', 'orange', 'pineapple']



```python
#Sort the list numerically:

thislist = [100, 50, 65, 82, 23]
thislist.sort()
print(thislist)
```

    [23, 50, 65, 82, 100]


#### Sort Descending
To sort descending, use the keyword argument reverse = True:


```python
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort(reverse = True)
print(thislist)
```

    ['pineapple', 'orange', 'mango', 'kiwi', 'banana']


### Python - List methods

|Method|Description|
|--- |--- |
|append()|Adds an element at the end of the list|
|clear()|Removes all the elements from the list|
|copy()|Returns a copy of the list|
|count()|Returns the number of elements with the specified value|
|extend()|Add the elements of a list (or any iterable), to the end of the current list|
|index()|Returns the index of the first element with the specified value|
|insert()|Adds an element at the specified position|
|pop()|Removes the element at the specified position|
|remove()|Removes the item with the specified value|
|reverse()|Reverses the order of the list|
|sort()|Sorts the list|



```python
(1, 2, 3)              < (1, 2, 4)
[1, 2, 3]              < [1, 2, 4]
'ABC' < 'C' < 'Pascal' < 'Python'
(1, 2, 3, 4)           < (1, 2, 4)
(1, 2)                 < (1, 2, -1)
(1, 2, 3)             == (1.0, 2.0, 3.0)
(1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)
```




    True




```python
(1, 2, 3) ==  (1, 2, 3)
```




    True



## Python - Classes and Object

Python is an object oriented programming language.

Almost everything in Python is an object, with its properties and methods.

A Class is like an object constructor, or a "blueprint" for creating objects.


```python
class myClass:
  x = 5

y = myClass()
print(y.x)
```

    5


### The __init__() Function

The examples above are classes and objects in their simplest form, and are not really useful in real life applications.

To understand the meaning of classes we have to understand the built-in **\__init__()** function.

All classes have a function called **\__init__()**, which is always executed when the class is being initiated.

Use the **\__init__()** function to assign values to object properties, or other operations that are necessary to do when the object is being created:


```python
#Create a class named Person, use the __init__() function to assign values for name and age:
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
  
  def introduction(self):
    print(f"my name is {self.name}, my age is {self.age}")

person = Person("Rambo", 25)

print(person.introduction())
```

    my name is Rambo, my age is 25
    None


Note: The **\__init__()** function is called automatically every time the class is being used to create a new object.

#### Object Methods
Objects can also contain methods. Methods in objects are functions that belong to the object.


```python
# Insert a function that prints a greeting, and execute it on the p1 object:
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def myfunc(self):
    print(f"Hello my name is {self.name}")

p1 = Person("John", 36)
p1.myfunc()
```

    Hello my name is John


> **Note**: The **self** parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.

#### The self Parameter

The **self** parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.

It does not have to be named **self** , you can call it whatever you like, but it has to be the first parameter of any function in the class:


```python
class Person:
  def __init__(somethingelse, name, age):
    somethingelse.name = name
    somethingelse.age = age
  
  def introduction(intro):
    print(f"My name is {intro.name}, and my age is {intro.age}")


person = Person("Hulk", 233)

person.introduction()
```

    My name is Hulk, and my age is 233


### Modify Object Properties
You can modify properties on objects like this:


```python
person.name = "Batman"
person.introduction()
```

    My name is Batman, and my age is 233


### Python - Inheritance

Inheritance allows us to define a class that inherits all the methods and properties from another class.

**Parent class** is the class being inherited from, also called base class.

**Child class** is the class that inherits from another class, also called derived class


```python
class Person:
  # constructor
  def __init__(self, name, age):
    self.name = name
    self.age = age
  
  def intorduction(self):
    print(f"My name is {self.name}, and my age is {self.age}")

class Student(Person):
  # constructor
  def __init__(self, name, age, gender):
    super().__init__(name, age)
    self.gender = gender
  
  def introduction(self):
    print(f"My name is {self.name}, my age is {self.age} and I am a {self.gender}")


person = Person("Hulk", 343)
person.intorduction()

stu = Student("Wonder women", 550, "Girl")
stu.introduction()

```

    My name is Hulk, and my age is 343
    My name is Wonder women, my age is 550 and I am a Girl


#### Python Iterators
An iterator is an object that contains a countable number of values.

An iterator is an object that can be iterated upon, meaning that you can traverse through all the values.

Technically, in Python, an iterator is an object which implements the iterator protocol, which consist of the methods **\_\_iter__()** and **\_\_next__()**.

#### Iterator vs Iterable
Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.

All these objects have a **iter()** method which is used to get an iterator:


```python
myTuple = ["apple", "banana", "cherry"]
myIter = iter(myTuple)
print(next(myIter))
print(next(myIter))
print(next(myIter))

```

    apple
    banana
    cherry


Even strings are iterable objects, and can return an iterator:


```python
mystr = "table"
strIter = iter(mystr)
print(next(strIter))
print(next(strIter))
print(next(strIter))
print(next(strIter))
print(next(strIter))
```

    t
    a
    b
    l
    e


### Looping Through an Iterator

We can also use a **for** loop to iterate through an iterable object:


```python
myList = ["apple", "banana", "cherry"]
myIter = iter(myList)

for item in myIter:
  print(item)
```

    apple
    banana
    cherry


The **for** loop actually creates an iterator object and executes the next() method for each loop.

### Create an Iterator
To create an object/class as an iterator you have to implement the methods **\_\_iter__()** and **\_\_next__()** to your object.

As you have learned in the Python Classes/Objects chapter, all classes have a function called **\_\_init__()**, which allows you to do some initializing when the object is being created.

The **\_\_iter__()** method acts similar, you can do operations (initializing etc.), but must always return the iterator object itself.

The **\_\_next__()** method also allows you to do operations, and must return the next item in the sequence.


```python
class MyIterator:
  def __iter__(self):
    self.a = 1
    return self
  
  def __next__(self):
    x = self.a
    self.a += 1
    return x

myIterator = MyIterator()
customIter = iter(myIterator)

print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
print(next(customIter))
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11


### StopIteration
The example above would continue forever if you had enough next() statements, or if it was used in a for loop.

To prevent the iteration to go on forever, we can use the **StopIteration** statement.

In the **\_\_next__()** method, we can add a terminating condition to raise an error if the iteration is done a specified number of times:


```python
class CustomIterator:
  def __iter__(self):
    self.a = 1
    return self
  
  def __next__(self):
    if self.a <= 20:
      x = self.a
      self.a += 1
      return x
    else:
      raise StopIteration

custIt = CustomIterator()
ci = iter(custIt)

for i in ci:
  print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20


### Python - Exception Handling 
The **try** block lets you test a block of code for errors.

The **except** block lets you handle the error.

The **finally** block lets you execute code, regardless of the result of the try- and except blocks.


```python
try:
  print(hello)
except:
  print("An exception occured")
```

    An exception occured


#### Many Exceptions
You can define as many exception blocks as you want, e.g. if you want to execute a special block of code for a special kind of error:


```python
try:
  print(hello)
except NameError:
  print("A name error happened")
except:
  print("Some error happened")
```

    A name error happened


#### Else
You can use the **else** keyword to define a block of code to be executed if no errors were raised:


```python
try:
  print("Hello Superman")
except:
  print("An exception occured")
else:
  print("No exception occured code running fine")
```

    Hello Superman
    No exception occured code running fine


#### Finally
The **finally** block, if specified, will be executed regardless if the try block raises an error or not.


```python
try:
  print(hello)
except:
  print("An error has occured")
finally:
  print("The 'try and excpet' block completed")
```

    An error has occured
    The 'try and excpet' block completed


This can be useful to close objects and clean up resources:

#### Raise an exception
As a Python developer you can choose to throw an exception if a condition occurs.

To throw (or raise) an exception, use the raise keyword.


```python
x = 5

if x > 6:
  raise Exception("Number isn't greater than 6")
```

## Chaining Method in python


```python
class ChainingClass:

  def __init__(self, data = ''):
    self.__obj = data
  
  def __repr__(self):
    return f"{self.__obj}"
  
  # First Method for chaining
  def toList(self):
    return ChainingClass(self.__obj.split(","))

  def toString(self):
    return ChainingClass(",".join(self.__obj))
  
  # Second Method to chaining
  def toList1(self):
    self.__obj = self.__obj.split(",")
    return self

  def toString1(self):
    self.__obj = ",".join(self.__obj)
    return self

if __name__ == "__main__":
  
  st1 = "1,2,3,4,5"
  d1 = ChainingClass(st1)
  print(d1.toList().toString())
  print(d1.toList1().toString1())

  
  
```

    1,2,3,4,5
    1,2,3,4,5




### Python - Magical or Dunder methods

Magic methods in Python are the special methods that start and end with the double underscores. They are also called dunder methods. Magic methods are not meant to be invoked directly by you, but the invocation happens internally from the class on a certain action. For example, when you add two numbers using the + operator, internally, the **\_\_add__()** method will be called.

Built-in classes in Python define many magic methods. Use the **dir()** function to see the number of magic methods inherited by a class. For example, the following lists all the attributes and methods defined in the **int** class.


```python
dir(int)
```




    ['__abs__',
     '__add__',
     '__and__',
     '__bool__',
     '__ceil__',
     '__class__',
     '__delattr__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__float__',
     '__floor__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__index__',
     '__init__',
     '__init_subclass__',
     '__int__',
     '__invert__',
     '__le__',
     '__lshift__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rlshift__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__round__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__trunc__',
     '__xor__',
     'as_integer_ratio',
     'bit_length',
     'conjugate',
     'denominator',
     'from_bytes',
     'imag',
     'numerator',
     'real',
     'to_bytes']



### repr() vs str() in Python

Apparently, there seems to be no difference between behavior of the **\_\_str__()** and **\_\_repr__()**. However, if we take a string object, the difference will be evident.


```python
x = "Hello Himanshu"
print(x.__str__())
print(x.__repr__())
```

    Hello Himanshu
    'Hello Himanshu'


Output of **\_\_repr__()** is in quotes whereas that of **\_\_str__()** is not. The reason can be traced to official definitions of these functions, which says that **\_\_repr__()** method and hence (repr() function) computes official string representation of an object. The str() function i.e. **\_\_str__()** method returns an informal or printable string representation of concerned object, which is used by the print() and format() functions.

### Acces Modifiers - Python - Public, Protected, Private Members

#### Public Members

Public members (generally methods declared in a class) are accessible from outside the class. The object of the same class is required to invoke a public method. This arrangement of private instance variables and public methods ensures the principle of data encapsulation.

All members in a Python class are public by default. Any member can be accessed from outside the class environment.


```python
# Example of Public

class Student:
  schoolName = "Ram Narayan Aryan"

  def __init__(self, name, age):
    self.name = name
    self.age = age

stu = Student("Rahul", 12)

display(stu.schoolName, stu.name, stu.age)
```


    'Ram Narayan Aryan'



    'Rahul'



    12


#### Protected Members

Protected members of a class are accessible from within the class and are also available to its sub-classes. No other environment is permitted access to it. This enables specific resources of the parent class to be inherited by the child class.

Python's convention to make an instance variable **protected** is to add a prefix _ (single underscore) to it. This effectively prevents it from being accessed unless it is from within a sub-class.


```python
class Student:
  _schoolName = "Aryan School of commerce"

  def __init__(self, name, age):
    self.name = name
    self._age = age

stu1 = Student("Radhe", 14)
print(stu1._schoolName)
print(stu1._age)
```

    Aryan School of commerce
    14


In fact, this doesn't prevent instance variables from accessing or modifying the instance. You can still perform the following operations:

However, you can define a property using property decorator and make it protected, as shown below.


```python
class Student:

  def __init__(self, name):
    self._name = name
  
  @property
  def name(self):
    return self._name
  
  @name.setter
  def name(self, newname):
    self._name = newname


stu2 = Student("Rahul")

stu2._name = "Mohan"
stu2._name



```




    'Mohan'



In python there is no such as protected and private but we use convention single _ for protected and __ for private so that if any developer seeing this he able to understand he doesn't need to play with those methods or class

### Python *args and **kwargs

In Python, we can pass a variable number of arguments to a function using special symbols. There are two special symbols:

\*args (Non Keyword Arguments)
**kwargs (Keyword Arguments)

We use \*args and \**kwargs as an argument when we are unsure about the number of arguments to pass in the functions.


```python
# Use of *args

def adder(*nums):
  result = 0

  for item in nums:
    result += item
  return result

print(adder(2))
print(adder(2,4))
print(adder(2,4, 8))
```

    2
    6
    14


#### Python \**kwargs
Python passes variable length non keyword argument to function using *args but we cannot use this to pass keyword argument. For this problem Python has got a solution called **kwargs, it allows us to pass the variable length of keyword arguments to the function.

In the function, we use the double asterisk ** before the parameter name to denote this type of argument. The arguments are passed as a dictionary and these arguments make a dictionary inside function with name same as the parameter excluding double asterisk **.


```python
def intro(**data):
  print("Checking the data format", type(data))

  for key, value in data.items():
    print(key, value)

intro(name="Kewal", age = 20, college = "Aryan")
```

    Checking the data format <class 'dict'>
    name Kewal
    age 20
    college Aryan


# End
