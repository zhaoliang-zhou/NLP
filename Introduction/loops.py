
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


# Function to return the sum of the first n positive odd numbers
# n: the number of initial odd numbers to sum
# Returns: sum as an integer
def sum_odd(n):
    # Your code here
    # set initial values
    cur = 1 # current value
    sum = 0 # sum of odd numbers
    i = 0 # iterations
    while i < n:
        sum = sum + cur # update sum at current step
        cur = cur + 2 # next odd number
        i = i + 1 # continue loop
    return sum

# Function to calculate the sum of the first N Fibonacci numbers
# n: the number of initial Fibonacci numbers to sum
# Returns: sum as an integer
def sum_fib(n):
    # Your code here
    fib = [0]*(n) # set a sequence of 0s empty list
    fib[0] = 0 # set initial value of fib seq. 1st element 0
    fib[1] = 1 # second from the fib seq is 1
    sum = fib[0] + fib[1] # initial fib sum
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
        sum = sum + fib[i]
    #print(fib) # check fib sequence
    return sum


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python loops.py
# It should produce the following output (with correct solution):
# 	    $ python loops.py
#       The sum of the first 5 positive odd numbers is: 25
#       The sum of the first 5 fibonacci numbers is: 7

def main():
    # Call the function to calculate sum
    osum = sum_odd(5) 

    # Print it out
    print(f'The sum of the first 5 positive odd numbers is: {osum}')

    # Call the function to calculate sum of fibonacci numbers
    fsum = sum_fib(5)
    print(f'The sum of the first 5 fibonacci numbers is: {fsum}')

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
