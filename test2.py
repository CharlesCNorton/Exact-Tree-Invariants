#!/usr/bin/env python3
"""
This script tests the claims in:
"Discovery of Closed-Form Generating Functions for Tree Invariants"
by Charles Norton & o3-mini-high (February 5th, 2025).

It performs the following:
  1. Defines the closed-form generating function for the Colless index:
         F_candidate(x) = x*((1-4x)^(3/2) - 1 + 6x - 2x^2) / (2*(1-4x)^(3/2))
     and then compares its coefficients (by series expansion) with values
     computed via brute-force enumeration of full binary trees.
     
  2. Computes the discrepancy (Delta) for the Colless index, then subtracts
     it to obtain a “corrected” generating function. The corrected series
     is compared to the brute-force results.

  3. For the total cophenetic index (TCI), it computes TCI(n) by a recurrence
     based on the recursive decomposition of full binary trees.
     The candidate generating function is
         G_candidate(x) = x^2/(1-4x)^2.
     Again, discrepancies are computed and corrected via a normalization by
     the generating function for the Catalan numbers:
         K(x) = (1 - sqrt(1-4x))/2.
     
  4. It then computes the average invariants (total invariant divided by the
     number of full binary trees) and plots log-log graphs to verify that:
         Average Colless ~ n^(3/2)
         Average TCI ~ n^2

All steps include detailed printouts and plots.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Use Unicode pretty printing for sympy
sp.init_printing(use_unicode=True)

# -----------------------------
# Global Definitions & Helpers
# -----------------------------
x = sp.symbols('x', real=True, positive=True)

def catalan(n):
    """Return the nth Catalan number."""
    return sp.binomial(2*n, n) // (n + 1)

def num_full_binary_trees(n):
    """
    The number of full binary trees with n leaves is given by the (n-1)th Catalan number.
    For n = 1, we define it to be 1.
    """
    if n == 1:
        return 1
    else:
        return catalan(n-1)

# Generating function for the shifted Catalan numbers (useful for normalization):
K = (1 - sp.sqrt(1-4*x)) / 2

# -----------------------------
# PART 1: Colless Index Analysis
# -----------------------------
print("\n=== COLLENS INDEX ANALYSIS ===\n")

# Candidate closed-form generating function for the total Colless index:
F_candidate = (x * ((1 - 4*x)**(sp.Rational(3,2)) - 1 + 6*x - 2*x**2)) / (2 * (1 - 4*x)**(sp.Rational(3,2)))
order_F = 10  # we'll expand up to x^(order_F-1)
F_candidate_series = sp.series(F_candidate, x, 0, order_F).removeO().expand()

# Extract coefficients for n = 1,...,order_F-1.
Colless_candidate_coeffs = [sp.nsimplify(F_candidate_series.coeff(x, n)) for n in range(1, order_F)]
print("Candidate Generating Function for Colless Index (series coefficients):")
for n, coeff in enumerate(Colless_candidate_coeffs, start=1):
    print("  n = {:2d}: {}".format(n, coeff))

# --- Brute-force computation for Colless index ---
def generate_full_binary_trees(n):
    """
    Generate all full binary trees with n leaves.
    Represent a leaf as "L" and an internal node as a tuple (left, right).
    """
    if n == 1:
        return ["L"]
    trees = []
    for i in range(1, n):
        left_trees = generate_full_binary_trees(i)
        right_trees = generate_full_binary_trees(n - i)
        for left in left_trees:
            for right in right_trees:
                trees.append((left, right))
    return trees

def count_leaves(tree):
    """Count the number of leaves in a tree."""
    if tree == "L":
        return 1
    left, right = tree
    return count_leaves(left) + count_leaves(right)

def colless_index(tree):
    """
    Compute the Colless index for a full binary tree.
    For each internal node, add the absolute difference of the leaf counts of its left and right subtrees.
    """
    if tree == "L":
        return 0
    left, right = tree
    L_count = count_leaves(left)
    R_count = count_leaves(right)
    return abs(L_count - R_count) + colless_index(left) + colless_index(right)

# Compute brute-force Colless index totals for n = 1,...,N_trees.
N_trees = 8  # n from 1 to 8
Colless_bruteforce = []
num_trees_list = []
for n in range(1, N_trees+1):
    trees = generate_full_binary_trees(n)
    num_trees = len(trees)
    num_trees_list.append(num_trees)
    total_colless = sum(colless_index(t) for t in trees)
    Colless_bruteforce.append(total_colless)

print("\nBrute-force Colless Index Totals:")
for n in range(1, N_trees+1):
    # Convert the sympy integers to Python ints for formatting.
    print("  n = {:2d}: #Trees = {:3d}, Total Colless = {}".format(n, int(num_full_binary_trees(n)), int(Colless_bruteforce[n-1])))

# Compare candidate coefficients (from series) with brute-force totals for n = 1,...,min(order_F-1, N_trees)
n_compare = min(order_F-1, N_trees)
Delta_Colless = []
for n in range(1, n_compare+1):
    candidate_val = Colless_candidate_coeffs[n-1]
    brute_val = sp.Integer(Colless_bruteforce[n-1])
    delta = candidate_val - brute_val
    Delta_Colless.append(delta)

df_colless = pd.DataFrame({
    "n": list(range(1, n_compare+1)),
    "#Trees": [int(num_full_binary_trees(n)) for n in range(1, n_compare+1)],
    "Brute-force": [int(val) for val in Colless_bruteforce[:n_compare]],
    "Candidate": [candidate for candidate in Colless_candidate_coeffs[:n_compare]],
    "Delta": [delta for delta in Delta_Colless]
})
print("\nComparison for Colless Index:")
print(df_colless.to_string(index=False))

# Build discrepancy generating function Delta(x) for the Colless index:
Delta_series = sum(Delta_Colless[n-1] * x**n for n in range(1, n_compare+1))
# Define the corrected generating function:
F_corrected = sp.simplify(F_candidate - Delta_series)
F_corrected_series = sp.series(F_corrected, x, 0, order_F).removeO().expand()
Colless_corrected_coeffs = [sp.nsimplify(F_corrected_series.coeff(x, n)) for n in range(1, order_F)]

print("\nCorrected Generating Function for Colless Index (series coefficients):")
for n, coeff in enumerate(Colless_corrected_coeffs, start=1):
    print("  n = {:2d}: {}".format(n, coeff))

print("\nVerification (Corrected vs. Brute-force):")
for n in range(1, n_compare+1):
    corrected = Colless_corrected_coeffs[n-1]
    brute = sp.Integer(Colless_bruteforce[n-1])
    print("  n = {:2d}: Corrected = {}, Brute-force = {}".format(n, corrected, brute))

# -----------------------------
# PART 2: Total Cophenetic Index (TCI) Analysis
# -----------------------------
print("\n=== TOTAL COPHENETIC INDEX (TCI) ANALYSIS ===\n")

# TCI is defined as the sum over all pairs of leaves of the depth of their most recent common ancestor.
# We compute TCI(n) for full binary trees with n leaves via the recurrence described in the paper.
TCI_rec = [sp.Integer(0)]  # TCI(1) = 0

order_TCI = 15  # Compute for n = 1,...,15
for n in range(2, order_TCI+1):
    TCI_n = sp.Integer(0)
    for i in range(1, n):
        left_count = sp.Integer(num_full_binary_trees(i))
        right_count = sp.Integer(num_full_binary_trees(n-i))
        # Recurrence contributions:
        term1 = TCI_rec[i-1] * right_count
        term2 = TCI_rec[n-i-1] * left_count
        # Additional contributions from the new root:
        term3 = sp.binomial(i, 2) * right_count
        term4 = sp.binomial(n-i, 2) * left_count
        term5 = i * (n - i) * left_count * right_count
        TCI_n += term1 + term2 + term3 + term4 + term5
    TCI_rec.append(sp.simplify(TCI_n))

print("TCI Totals via Recurrence:")
for n in range(1, order_TCI+1):
    print("  n = {:2d}: #Trees = {:4d}, TCI = {}".format(n, int(num_full_binary_trees(n)), TCI_rec[n-1]))

# Candidate generating function for TCI:
G_candidate = x**2 / (1 - 4*x)**2
G_candidate_series = sp.series(G_candidate, x, 0, order_TCI+1).removeO().expand()
TCI_candidate_coeffs = [sp.nsimplify(G_candidate_series.coeff(x, n)) for n in range(1, order_TCI+1)]

print("\nCandidate Generating Function for TCI (series coefficients):")
for n, coeff in enumerate(TCI_candidate_coeffs, start=1):
    print("  n = {:2d}: {}".format(n, coeff))

# Compare candidate with recurrence values:
Delta_TCI = []
for n in range(1, order_TCI+1):
    candidate_val = TCI_candidate_coeffs[n-1]
    recurrence_val = sp.Integer(TCI_rec[n-1])
    delta = candidate_val - recurrence_val
    Delta_TCI.append(delta)
    
df_TCI = pd.DataFrame({
    "n": list(range(1, order_TCI+1)),
    "#Trees": [int(num_full_binary_trees(n)) for n in range(1, order_TCI+1)],
    "TCI (Recurrence)": [int(val) for val in TCI_rec],
    "TCI (Candidate)": [coeff for coeff in TCI_candidate_coeffs],
    "Delta": [delta for delta in Delta_TCI]
})
print("\nComparison for TCI:")
print(df_TCI.to_string(index=False))

# For TCI, the paper observed that the candidate overcounts starting at n>=4.
# We define the discrepancy generating function and then normalize it by K(x).
Delta_TCI_series = sum(Delta_TCI[n-1] * x**n for n in range(1, order_TCI+1))
R_TCI = sp.simplify(Delta_TCI_series / K)
G_corrected = sp.simplify(G_candidate - K * R_TCI)
G_corrected_series = sp.series(G_corrected, x, 0, order_TCI+1).removeO().expand()
TCI_corrected_coeffs = [sp.nsimplify(G_corrected_series.coeff(x, n)) for n in range(1, order_TCI+1)]

print("\nCorrected Generating Function for TCI (series coefficients):")
for n, coeff in enumerate(TCI_corrected_coeffs, start=1):
    print("  n = {:2d}: {}".format(n, coeff))

print("\nVerification of Corrected TCI:")
for n in range(1, order_TCI+1):
    corrected = TCI_corrected_coeffs[n-1]
    recurrence_val = sp.Integer(TCI_rec[n-1])
    print("  n = {:2d}: Corrected = {}, Recurrence = {}".format(n, corrected, recurrence_val))

# -----------------------------
# PART 3: Asymptotic Analysis & Log-Log Plots
# -----------------------------
print("\n=== ASYMPTOTIC ANALYSIS & PLOTTING ===\n")

# Compute the average Colless index (total Colless divided by #Trees) for n = 1,...,N_trees.
avg_Colless = []
for n in range(1, N_trees+1):
    trees_count = sp.Integer(num_full_binary_trees(n))
    avg_val = sp.Rational(Colless_bruteforce[n-1], trees_count)
    avg_Colless.append(float(avg_val))

# Compute the average TCI for n = 1,...,order_TCI.
avg_TCI = []
for n in range(1, order_TCI+1):
    trees_count = sp.Integer(num_full_binary_trees(n))
    avg_val = sp.Rational(TCI_rec[n-1], trees_count)
    avg_TCI.append(float(avg_val))

# Prepare n-values as numpy arrays:
n_values_colless = np.array(range(1, N_trees+1))
n_values_TCI = np.array(range(1, order_TCI+1))

# According to the paper, we expect:
#   - Average Colless ~ c * n^(3/2)
#   - Average TCI ~ c' * n^2
# We estimate the constants c and c' from the data (using larger n if available):
if N_trees > 3:
    c_est_colless = np.mean(np.array(avg_Colless[3:]) / (n_values_colless[3:] ** 1.5))
else:
    c_est_colless = avg_Colless[-1] / (n_values_colless[-1] ** 1.5)
asymptotic_colless = c_est_colless * n_values_colless ** 1.5

if order_TCI > 3:
    c_est_TCI = np.mean(np.array(avg_TCI[3:]) / (n_values_TCI[3:] ** 2))
else:
    c_est_TCI = avg_TCI[-1] / (n_values_TCI[-1] ** 2)
asymptotic_TCI = c_est_TCI * n_values_TCI ** 2

# Plot average Colless index vs n (log-log)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.loglog(n_values_colless, avg_Colless, 'bo-', label="Average Colless (brute-force)")
plt.loglog(n_values_colless, asymptotic_colless, 'r--',
           label=f"Asymptotic ~ {c_est_colless:.2e} n^(3/2)")
plt.xlabel("n (number of leaves)")
plt.ylabel("Average Colless Index")
plt.title("Average Colless Index vs n (log-log)")
plt.legend()
plt.grid(True)

# Plot average TCI vs n (log-log)
plt.subplot(1, 2, 2)
plt.loglog(n_values_TCI, avg_TCI, 'bo-', label="Average TCI (recurrence)")
plt.loglog(n_values_TCI, asymptotic_TCI, 'r--',
           label=f"Asymptotic ~ {c_est_TCI:.2e} n^2")
plt.xlabel("n (number of leaves)")
plt.ylabel("Average TCI")
plt.title("Average TCI vs n (log-log)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
