def pair_sum(arr, k):
  temp_dict = {}

  for i in arr:
    if k-i in temp_dict:
      return (i, k-i)
    else:
      temp_dict[i] = 1
  return False


arr = [1,3,5,9]
k = 12

print(pair_sum(arr, k))