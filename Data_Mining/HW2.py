# %%

## HOME WORK 1 SOLUTIONS

# Run each cell below to find the solution

# %%
# Import required libraries
import numpy as np

# %%
# Question 1
print("""
1. Given an integer, n, say (0 < n < 100), perform the following conditional actions:
   • If n is odd, print 'Weird'
   • If n is even and in the inclusive range of 2 to 5, print 'Not Weird'
   • If n is even and in the inclusive range of 6 to 20, print 'Weird'
   • If n is even and greater than 20, print 'Not Weird'
""")


def check_condition(n):

    # Base case
    if n % 2 != 0:
        print("Weird")
    # Condition 2
    elif 2 <= n <= 5:
        print("Not Weird")
    # Condition 3
    elif 6 <= n <= 20:
        print("Weird")
    # Condition 4
    elif n > 20:
        print("Not Weird")


n = np.random.randint(1, 100)  # Generate random numbers between 1 to 100
check_condition(n)

# %%
# Question 2

print("""
2. Write a function what computes the maximum between two scalars. 
Convert that function to work on two arrays. HINT: use numpy.vectorize
""")

def max_scalar(a, b):

    '''
    Function to compute max of two scalars
    Input:
    a - array 1
    b - array 2
    Output: 
    maximum of a and b
    '''
    return a if a > b else b 

# Vectorize max scalar function
vectorized = np.vectorize(max_scalar) 

x = np.array([1, 2, 3]) # array 1
y = np.array([4, 5, 6]) # array 2

output = vectorized(x, y)
print('Maximum between two arrays is :',output)

# %%
# Question 3

print("""
3. Generate 100 random variables which follow normal distribution with mean 2 and standard deviation 5. 
Compute their mean, median and variance. HINT use numpy.random.normal, numpy.median
""")
# You can se np.random.normal to create normally distributed random sample 
# with custom mean and sd
input = np.random.normal(loc=2, scale=5, size=100)

# Calculate mean, median, and variance
mean = np.mean(input)
median = np.median(input)
variance = np.var(input)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {variance}")



# %%
# Question 4

print("""
4.	Insert np.nan values at 20 random positions into the array from the previous exercise 
HINT use np.random.choice
""")
# Generate random 20 positions to insert nan values in the input array
indices = np.random.choice(range(100), size=20, replace=False)

# Replace indices in the array with np.nan
input[indices] = np.nan

print("Array with nan values:")
print(input)

# %%
# Question 5

print("""
5. Find the number and position of missing values in the array from the above exercise. 
HINT use  np.where, np.isnan
""")

# store the indices with nan values
nan_indices = np.where(np.isnan(input))[0]

# Count the number of missing values
nan_count = len(nan_indices)

print(f"Number of missing values: {nan_count}")
print(f"Positions of missing values: {nan_indices}")

# %%
# Question 6

print("""
6. Reshape the array with missing values to be a 10 by 10 2d array. 
""")

# Reshape array with missing values to be 10 by 10 2d array with reshape function
reshaped_array = input.reshape(10, 10)

print(reshaped_array)

# %%
# Question 7

print("""
7. Drop rows that contain a missing value from 2d e array resulted in problem 6. 
""")

# Drop rows with any np.nan values
cleaned_array = reshaped_array[~np.isnan(reshaped_array).any(axis=1)]

print(cleaned_array)

# %%
# Question 8

print("""
8. Generate 1d array of 20 Poisson random variables with mean 10.
""")

poisson_variables = np.random.poisson(lam=10, size=20)

print(poisson_variables)

# %%
# Question 9

print("""
9. Rank the values in the array from problem  8 from the smallest to the largest
""")

# sort the values from smallest to largest to rank them
rank = np.argsort(poisson_variables)
ranked_array = poisson_variables[rank]

print(ranked_array)

# %%
# Question 10

print("""
10. Perform min-max transformation with elements of the array from problem  
9 so that all values will be inside [0,1] interval. 
""")

# min-max transformation
minimum = np.min(ranked_array)
maximum = np.max(ranked_array)
normalized_array = (ranked_array - minimum) / (maximum - minimum)

print(normalized_array)

# %%
