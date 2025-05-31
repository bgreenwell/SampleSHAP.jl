function compute_square_sum(arr)
    # Initialize total to zero for summation
    total = 0
    # Iterate over each element in the array
    for x in arr
        # Add the square of the element to the total
        total += x^2
    end
    return total
end