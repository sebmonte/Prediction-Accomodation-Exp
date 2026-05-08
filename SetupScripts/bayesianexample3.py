import numpy as np
import matplotlib.pyplot as plt

def plot_decomposition():
    # Fix world counts
    N1 = 1e9   # stringent worlds
    N2 = 1e4   # lax worlds

    # Fix lax life-density
    p_lax = 1.0  # D2/N2

    # Vary stringent life-density
    p_str_values = np.linspace(1e-8, 0.5, 300)

    BF_stringency_vals = []
    BF_life_given_str_vals = []
    BF_total_vals = []

    for p_str in p_str_values:
        D1 = p_str * N1
        D2 = p_lax * N2

        # --- BF for stringency ---
        BF_stringency = (D1 / (D1 + D2)) / (N1 / (N1 + N2))

        # --- BF for life given stringency ---
        BF_life_given_str = 1 / (D1 / N1)

        # --- Total BF ---
        BF_total = BF_stringency * BF_life_given_str

        BF_stringency_vals.append(BF_stringency)
        BF_life_given_str_vals.append(BF_life_given_str)
        BF_total_vals.append(BF_total)

    # --- Plot ---
    plt.figure(figsize=(8, 5))

    plt.plot(p_str_values, BF_stringency_vals, label="BF stringency (against design)")
    plt.plot(p_str_values, BF_life_given_str_vals, label="BF life | stringency (for design)")
    plt.plot(p_str_values, BF_total_vals, linestyle="--", label="Total BF")

    plt.yscale("log")
    plt.axhline(1, linestyle=":", color="black")

    plt.xlabel("Life-density in stringent worlds (D1 / N1)")
    plt.ylabel("Bayes Factor")
    plt.title("Decomposition: Stringency vs Life Evidence")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_decomposition()