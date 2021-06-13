
# Sorting array
class Sort:
    # Selection Sort -> O(n2)
    def selectionSort(self, arr):
        for i in range(len(arr[:-2])):
            min = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[min]:
                    min = j

            arr[i], arr[min] = arr[min], arr[i]

        return arr

        # Bubble Sort -> O(n2)
    def bubbleSort(self, arr):
        for i in range(len(arr[:-1])):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]

        return arr

    # Insertion Sort -> O(n2)
    # arr = [64, 25, 12, 22, 11]
    def insertionSort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i-1
            while j >= 0 and key < arr[j]:
                arr[j+1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    # Quick Sort -> O(n2) worst case O(nlogn) average case
    # arr = [64, 25, 12, 22, 11]
    def quickSort(self, arr, low, high):
        if low < high:
            pi = self.partition(arr, low, high)
            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi+1, high)

        return arr

    def partition(self, arr, left, right):
        pivot = right
        l = left
        r = right - 1
        while l <= r:
            if arr[l] > arr[pivot] and arr[r] < arr[pivot]:
                arr[l], arr[r] = arr[r], arr[l]
                l += 1
                r -= 1
            if arr[l] < arr[pivot]:
                l += 1
            if arr[r] > arr[pivot]:
                r -= 1
        arr[l], arr[pivot] = arr[pivot], arr[l]
        return l

    # Merge Sort
    # arr = [64, 25, 12, 22, 11]

    def mergeSort(self, arr, left, right):
        if left < right:
            middle = (left + right)//2
            self.mergeSort(arr, left, middle)
            self.mergeSort(arr, middle+1, right)
            self.merge(arr, left, middle, right)
        return arr

    def merge(self, arr, left, middle, right):
        k = left

        i = j = 0
        l_arr = arr[left: middle + 1]
        r_arr = arr[middle+1: right+1]
        while i < len(l_arr) and j < len(r_arr):
            if l_arr[i] < r_arr[j]:
                arr[k] = l_arr[i]
                k += 1
                i += 1
            else:
                arr[k] = r_arr[j]
                k += 1
                j += 1

        while i < len(l_arr):
            arr[k] = l_arr[i]
            k += 1
            i += 1

        while j < len(r_arr):
            arr[k] = r_arr[j]
            k += 1
            j += 1

    # Heap Sort -> O(nlogn) worst case and average

    def heapSort(self, arr):
        # create max heap
        n = len(arr)
        for i in range(n//2-1, -1, -1):
            self.heapify(arr, n, i)

        for i in range(n-1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.heapify(arr, i, 0)

        return arr

    def heapify(self, arr, n, i):
        largest = i
        l = 2*i + 1
        r = 2*i + 2

        if l < n and arr[l] > arr[largest]:
            largest = l

        if r < n and arr[r] > arr[largest]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]

            self.heapify(arr, n, largest)


arr = [64, 25, 12, 22, 11, 4, -2, 0]
sort = Sort()
# print(sort.selectionSort(arr))
# print(sort.bubbleSort(arr))
# print(sort.insertionSort(arr))
# print(sort.quickSort(arr, 0, len(arr) - 1))
# print(sort.mergeSort(arr, 0, len(arr) - 1))
print(sort.heapSort(arr))
