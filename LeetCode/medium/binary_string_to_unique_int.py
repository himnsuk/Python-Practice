def count_unique_decimals(binary_string):
    def generate_subsequences(index, current):
        print(f"index -> {index}")
        print(f"Current -> {current}")
        if index == len(binary_string):
            if current:  # Only consider non-empty subsequences

                unique_numbers.add(int(current, 2))
            return
        
        # Include the current character
        generate_subsequences(index + 1, current + binary_string[index])
        
        # Exclude the current character
        generate_subsequences(index + 1, current)
    
    unique_numbers = set()
    generate_subsequences(0, "")
    return len(unique_numbers)

# Example Usage
binary_string = "011"
result = count_unique_decimals(binary_string)
print(f"Number of unique decimal numbers: {result}")
