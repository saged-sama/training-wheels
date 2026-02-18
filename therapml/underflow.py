import numpy as np

def lse(arr):
    max_val = np.max(arr)
    sum = max_val + np.log(np.sum(np.exp(arr - max_val)))
    return sum

def underflow_sim(n = 1000, low=0.0, high=1.0):
    arr = np.random.uniform(low=low, high=high, size=n)

    product = np.prod(arr)
    print(f"Product of {n} random numbers between {low} and {high}: {product}")

    log_sum = np.sum(np.log(arr))
    print(f"Sum of logs of {n} random numbers between {low} and {high}: {log_sum}")
    product_log = np.exp(log_sum)
    print(f"Exponential of the sum of logs: {product_log}")

    print("\n\n")

    sum = np.sum(arr)
    print(f"Sum of {n} random numbers between {low} and {high}: {sum}")

    log_sum_exp = lse(np.log(arr))
    print(f"Log-Sum-Exp of {n} random numbers between {low} and {high}: {log_sum_exp}")

    log_sum_exp_sum = np.exp(log_sum_exp)
    print(f"Exponential of Log-Sum-Exp: {log_sum_exp_sum}")

if __name__ == "__main__":
    underflow_sim(n=1000, low=1.0e-10, high=1.0e-5)