#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int sort_arr(int arr[], int temp[], int l, int r);
int merge(int arr[], int temp[], int l, int m, int r);

int merge_sort(int arr[], int array_size)
{
  int *temp = (int *)malloc(sizeof(int)*array_size);
  return sort_arr(arr, temp, 0, array_size - 1);
}

int sort_arr(int arr[], int temp[], int l, int r)
{
  int m, inv_count = 0;
  if (r > l)
  {
    m = (r + l)/2;

    inv_count = sort_arr(arr, temp, l, m);
    inv_count += sort_arr(arr, temp, m+1, r);

    inv_count += merge(arr, temp, l, m+1, r);
  }
  return inv_count;
}

int merge(int arr[], int temp[], int l, int m, int r)
{
  int i, j, k;
  int inv_count = 0;

  i = l;
  j = m;
  k = l;
  while ((i <= m - 1) && (j <= r))
  {
    if (arr[i] <= arr[j])
    {
      temp[k++] = arr[i++];
    }
    else
    {
      temp[k++] = arr[j++];

      inv_count = inv_count + (m - i);
    }
  }

  while (i <= m - 1)
    temp[k++] = arr[i++];

  while (j <= r)
    temp[k++] = arr[j++];

  for (i=l; i <= r; i++)
    arr[i] = temp[i];

  return inv_count;
}

int main(int argv, char** args)
{
  int arr[100000];
  int arr_len;
  std::cin >> arr_len;

  for(int i = 0; i < arr_len; i++){
    std::cin >> arr[i];
  }
  printf("%d", merge_sort(arr, arr_len));
  getchar();
  return 0;
}
