# Catalan numbers are a sequence of natural numbers that occurs in many interesting counting problems like following.

# Count the number of expressions containing n pairs of parentheses which are correctly matched. For n = 3, possible expressions are ((())), ()(()), ()()(), (())(), (()()).
# Count the number of possible Binary Search Trees with n keys (See this)
# Count the number of full binary trees (A rooted binary tree is full if every vertex has either two children or no children) with n+1 leaves.
# Given a number n, return the number of ways you can draw n chords in a circle with 2 x n points such that no 2 chords intersect.
# See this for more applications. 
# The first few Catalan numbers for n = 0, 1, 2, 3, … are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862,


# Brute-Force method 

# def catalan_number(n):
#     if n == 0 or n == 1:
#         return 1
#     res = 0

#     for i in range(n):
#         res += catalan_number(i)*catalan_number(n-i-1)
    
#     return res

# print(catalan_number(9))

# Dynamic Programming
# Time complexity O(n^2)

# def get_catalan_number(n, catalan):

#     res = 0
#     for i in range(2, n+1):
#         for j in range(i):
#             catalan[i] += catalan[j]*catalan[i-j-1]
    
#     print(catalan)
#     return catalan[n]




# if __name__ == "__main__":
#     n = 20
#     catalan = [1, 1] + [0]*(n-1)

#     print(get_catalan_number(n, catalan))


# Tweeking solution with O(n) complexity


def get_coefficient(n, k):
    if k < n-k:
        k = n-k
    
    res = 1
    for i in range(k):
        res *= n-i
        res //= i+1
    
    return res

def get_catalan(n):
    c = get_coefficient(2*n, n)
    return c//(n+1)

print(get_catalan(5))
