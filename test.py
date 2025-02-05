import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setup: Symbols and parameters
x = sp.symbols('x')
order = 40  # maximum n (number of leaves) to test

# Utility: Shifted Catalan numbers (number of full binary trees with n leaves)
def catalan(n):
    return sp.binomial(2*n, n) // (n+1)

# Create a list B such that B[n-1] is the number of full binary trees with n leaves
B = [1 if n == 1 else catalan(n-1) for n in range(1, order+1)]

# Part 1: Colless Index
F_colless = (x * ((1 - 4*x)**(sp.Rational(3,2)) - 1 + 6*x - 2*x**2)) / (2 * (1 - 4*x)**(sp.Rational(3,2)))
F_series = sp.series(F_colless, x, 0, order+1).expand()
Colless_coeffs = [sp.nsimplify(F_series.coeff(x, n)) for n in range(1, order+1)]
avg_Colless = [sp.N(Colless_coeffs[i] / B[i]) for i in range(order)]
avg_Colless = np.array([float(val) for val in avg_Colless])

# Part 2: Total Cophenetic Index (TCI)
TCI_rec = [0]  # TCI_rec[0] corresponds to n = 1
for n in range(2, order+1):
    TCI_val = 0
    for i in range(1, n):
        term1 = TCI_rec[i-1] * B[n-i-1]
        term2 = TCI_rec[n-i-1] * B[i-1]
        term3 = sp.binomial(i, 2) * B[n-i-1]
        term4 = sp.binomial(n-i, 2) * B[i-1]
        term5 = i * (n - i) * B[i-1] * B[n-i-1]
        TCI_val += term1 + term2 + term3 + term4 + term5
    TCI_rec.append(sp.simplify(TCI_val))

G_candidate = x**2 / (1 - 4*x)**2
G_series_TCI = sp.series(G_candidate, x, 0, order+1).expand()
TCI_candidate = [sp.nsimplify(G_series_TCI.coeff(x, n)) for n in range(1, order+1)]

Delta_TCI = [sp.nsimplify(TCI_candidate[i] - TCI_rec[i]) for i in range(order)]
Delta_series_TCI = sum(Delta_TCI[i] * x**(i+1) for i in range(order))

K = (1 - sp.sqrt(1 - 4*x)) / 2
R_TCI = sp.simplify(Delta_series_TCI / K)
G_corrected_TCI = sp.simplify(G_candidate - K * R_TCI)
G_corr_series_TCI = sp.series(G_corrected_TCI, x, 0, order+1).expand()
TCI_corrected = [sp.nsimplify(G_corr_series_TCI.coeff(x, n)) for n in range(1, order+1)]
avg_TCI = [sp.N(TCI_corrected[i] / B[i]) for i in range(order)]
avg_TCI = np.array([float(val) for val in avg_TCI])

# Part 3: Asymptotic Analysis and Log–Log Plots
n_vals = np.arange(1, order+1)
c_est = np.mean(avg_Colless[order//2:] / (n_vals[order//2:]**1.5))
asymptotic_Colless = c_est * n_vals**1.5

c_est_TCI = np.mean(avg_TCI[order//2:] / (n_vals[order//2:]**2))
asymptotic_TCI = c_est_TCI * n_vals**2

# Generating log-log plots
plt.figure(figsize=(12,6))

# Colless Index Plot
plt.subplot(1,2,1)
plt.loglog(n_vals, avg_Colless, 'bo-', label="Average Colless (closed-form)")
plt.loglog(n_vals, asymptotic_Colless, 'r--', label=f"Asymptotic ∼ {c_est:.2e}·n^(3/2)")
plt.xlabel("n (number of leaves)")
plt.ylabel("Average Colless Index")
plt.title("Average Colless Index vs n (log-log)")
plt.legend()
plt.grid(True, which="both", ls="--")

# TCI Plot
plt.subplot(1,2,2)
plt.loglog(n_vals, avg_TCI, 'bo-', label="Average TCI (closed-form)")
plt.loglog(n_vals, asymptotic_TCI, 'r--', label=f"Asymptotic ∼ {c_est_TCI:.2e}·n²")
plt.xlabel("n (number of leaves)")
plt.ylabel("Average Total Cophenetic Index")
plt.title("Average TCI vs n (log-log)")
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()

# Displaying results as DataFrames
colless_df = pd.DataFrame({
    "n": n_vals,
    "B(n)": B[:order],
    "I(n) (Colless)": Colless_coeffs,
    "Average Colless": avg_Colless
})

tci_df = pd.DataFrame({
    "n": n_vals,
    "B(n)": B[:order],
    "TCI(n) Recurrence": TCI_rec,
    "TCI_candidate": TCI_candidate,
    "TCI(n) Corrected": TCI_corrected,
    "Average TCI": avg_TCI
})

# Print first few rows of the tables
print("Colless Index Table:")
print(colless_df.head(10))

print("\nTotal Cophenetic Index Table:")
print(tci_df.head(10))
