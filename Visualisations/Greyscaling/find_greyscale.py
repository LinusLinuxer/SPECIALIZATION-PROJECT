import numpy as np
import itertools

# RGB values (in order R, G, B)
b = np.array([166, 183, 196])  # background
w = np.array([74, 72, 102])  # writing

# Fine resolution step
step = 0.01
coefficients = np.arange(0, 1 + step, step)

max_contrast = 0
best_combo = (0, 0, 0)

# Brute-force search over all valid (alpha, beta, gamma) triples
for alpha, beta in itertools.product(coefficients, repeat=2):
    gamma = 1 - alpha - beta
    if 0 <= gamma <= 1:
        # Compute grayscale values
        gray_b = alpha * b[0] + beta * b[1] + gamma * b[2]
        gray_w = alpha * w[0] + beta * w[1] + gamma * w[2]
        contrast = abs(gray_b - gray_w)

        if contrast > max_contrast:
            max_contrast = contrast
            best_combo = (alpha, beta, gamma)

# Output the result
print("Best coefficients (R, G, B):", best_combo)
print("Maximum grayscale contrast:", max_contrast)
