
import functools

random_prime = 127

def robin_karp(s, t):
    tn = len(t)
    sn = len(s)
    t_hash = 0
    # Generate Substring hash
    t_hash2 = functools.reduce(lambda h, c: ord(c)*(random_prime**h), t, 0)
    print(t_hash2)
    for i,x in enumerate(t):
        t_hash += ord(x)*(random_prime**i)


    for y in range(sn-tn+1):
        temp_hash = 0
        for key, val in enumerate(s[y:y+tn]):
            temp_hash += ord(val)*(random_prime**key)
        
        if temp_hash == t_hash:
            sub = s[y:y+tn]
            flag = True
            for k in range(tn):
                if sub[k] != t[k]:
                    flag = False
                    break
            if (flag):
                return True
    return False

s = "GEEKS FOR GEEKS"
t = "FOR"
print(robin_karp(s,t))