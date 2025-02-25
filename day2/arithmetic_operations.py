def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power(a, b):
    return a ** b

def mod(a, b):
    if b == 0:
        raise ValueError("Cannot mod by zero")
    return a % b

# Example usage
if __name__ == "__main__":

    x = 10
    y = 5

    print(f"{x} + {y} = ", add(x, y))
    print(f"{x} - {y} = ", subtract(x, y))
    print(f"{x} * {y} = ", multiply(x, y))
    print(f"{x} / {y} = ", divide(x, y))
    print(f"{x} ^ {y} = ", power(x, y))
    print(f"{x} % {y} = ", mod(x, y))