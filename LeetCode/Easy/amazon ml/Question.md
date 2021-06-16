Question

1

Max. score: 100.00

Unique numbers

Consider a sequence of integers as _1, 1, 2, 2, 3, 3,_ ... that satisfies the following properties:

*   The sequence is sorted
*   For every number x, where x≥1x \\ge 1, _x_ appears 2×⌊x⌋2 \\times \\lfloor \\sqrt x \\rfloor number of times

**Task**

Process _Q_ queries of type _(l, r)._ For each query, determine the number of unique integers appearing in the range from _l_ to _r_ (inclusive) in the sequence described in the problem statement.

_Note_s

*   ⌊n⌋\=y\\lfloor n \\rfloor = y , such that _y_ is the greatest integer less than or equal to _n_.
*   _1_\-based indexing is followed.

**Example**

_Assumptions_

*   _Q = 2_
*   _query = \[\[1 5\], \[2 20\]\]_

_Approach_

First 20 terms of the sequence are _1 1 2 2 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7_

For 1st query:

*   _l = 1_ and _r = 5_: There are three unique numbers _(1, 2, and 3)_ in the range _1_ to _5_.
*   Hence, the output is _3_.

For 2nd query:

*   _l = 2_ and _r = 20_: There are seven unique numbers _(1, 2, 3, 4, 5, 6, and 7)_ in the range _2_ to _20_.
*   Hence the output is _7_.

Therefore, the answer is _\[3 7\]._

**Function description**

Complete the _solve_ function provided in the editor. This function takes the following 2 parameters and returns an integer array representing the output of each query:

*   _Q_: Represents an integer denoting the number of queries
*   _query_: Represents an integer 2-D array _query_ of _Q_ rows and 2 columns describing each query.

**Input format**

**Note:** This is the input format that you must use to provide custom input (available above the **Compile and Test** button). 

*   The first line contains an integer _Q_ denoting the number of queries.
*   The next _Q_ lines contain two space-separated integers _l_ and _r_ as described in the problem statement.

**Output format**

Print _Q_ space-separated integers, representing the output for each query.

**Constraints**

**1≤Q≤1051 \\le Q \\le 10^5**

1≤l≤r≤1091 \\le l \\le r \\le 10^{9}

**Code snippets (also called starter code/boilerplate code)** 

This question has code snippets for C, CPP, Java, and Python.

Sample input 1

Copy

3

1 10

2 8

7 15

Sample output 1

Copy

4 4 3 

Explanation

_Given_

*   _Q = 3_
*   _query = \[\[1 10\], \[2 8\], \[7 15\]\]_

_Approach_

First _15_ terms of the sequence are _1 1 2 2 3 3 4 4 4 4 5 5 5 5 6_ 

For 1st query:

*   _l = 1_ and _r = 10_: There are four unique numbers _(1, 2, 3, and 4)_ in the range _1_ to _10_.
*   Hence, the output is _4_.

For 2nd query:

*   _l = 2_ and _r = 8_: There are four unique numbers _(1, 2, 3, and 4)_ in the range _2_ to _8_.
*   Hence, the output is _4_.

For 3rd query:

*   _l = 7_ and _r = 15_: There are three unique numbers _(4, 5, and 6)_ in the range _7_ to _15_.
*   Hence, the output is _3_.

Therefore, the answer is _\[4 4 3\]._

Note: Your code must be able to print the sample output from the provided sample input. However, your code is run against multiple hidden test cases. Therefore, your code must pass these hidden test cases to solve the problem statement.

Time Limit: 1.0 sec(s) for each input file

Memory Limit: 256 MB

Source Limit: 1024 KB

Marking Scheme: Score is assigned if any testcase passes

Allowed Languages: Python, Python 3, Python 3.8