# def longestCommonPrefix(strs):
#     # whole string match
#     # no match
#     if len(strs) < 1 or len(strs) > 200:
#         return 0
#     x = 0
#     pref = ""
#     while(x >= 0):
#         if len(strs[0]) >= x:
#             let = strs[0][x]
#             match = True
#             for st in strs:
#                 if st[x] != let:
#                     match = False
#                     x = -1
#                     break
#             if not match:
#                 break
#             pref += let
#             x += 1
#         else:
#             x = -1
#     return pref

# strs = ["flower","flow","flight"]
# print(longestCommonPrefix(strs))


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) < 1 or len(strs) > 200:
            return 0
        x = 0
        pref = ""
        while(x >= 0):
            if x < len(strs[0]):
                let = strs[0][x]
                match = True
                for st in strs:
                    if x > len(st):
                        match = False
                        x = -1
                        break
                    if st[x] != let:
                        match = False
                        x = -1
                        break
                if not match:
                    break
                pref += let
                x += 1
            else:
                x = -1
        return pref
