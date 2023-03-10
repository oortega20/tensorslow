
def sigmoid(x: float) -> float:
    return 1 / (1 + (math.e ** (-1 * x)))

def tanh(x: float) -> float:
    numerator = (math.e ** x) - math.e ** (-1 * x)
    denominator = (math.e ** x) + math.e ** (-1 * x)
    return numerator / denominator

def relu(x: float) -> float:
    return x if x > 0 else 0
 
