def generate_random_numbers(count, lower_bound=0, upper_bound=100):
    import random
    return [random.randint(lower_bound, upper_bound) for _ in range(count)]