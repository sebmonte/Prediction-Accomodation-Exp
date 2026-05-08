import numpy as np
import matplotlib.pyplot as plt


def compute_bfs(N_lax, N_str, p_lax, p_str):
    """
    N_lax: total worlds under lax laws
    N_str: total worlds under stringent laws
    p_lax: proportion of life-worlds in lax
    p_str: proportion of life-worlds in stringent

    Returns:
        BF_life, BF_lax
    """

    # Life counts
    D_lax = p_lax * N_lax
    D_str = p_str * N_str

    total_life = D_lax + D_str
    total_worlds = N_lax + N_str

    # --- BF for life ---
    p_life_given_non_design = total_life / total_worlds
    BF_life = 1 / p_life_given_non_design

    # --- BF for laxity ---
    p_lax_given_design = D_lax / total_life
    p_lax_given_non_design = N_lax / total_worlds

    BF_lax = p_lax_given_design / p_lax_given_non_design

    return BF_life, BF_lax


def plot_bfs():
    # --- Fixed parameters ---
    N_lax = 1e4        # total lax worlds
    N_str = 1e9        # total stringent worlds
    p_lax = 1        # lax always very life-friendly

    # Vary stringent life-density (always less than lax)
    p_str_values = np.linspace(1e-8, p_lax - 1e-3, 200)

    bf_life_vals = []
    bf_lax_vals = []

    for p_str in p_str_values:
        bf_life, bf_lax = compute_bfs(N_lax, N_str, p_lax, p_str)
        bf_life_vals.append(bf_life)
        bf_lax_vals.append(bf_lax)

    # --- Plot ---
    plt.figure(figsize=(8, 5))

    plt.plot(p_str_values, bf_life_vals, label="BF for Life")
    plt.plot(p_str_values, bf_lax_vals, linestyle='--', label="BF for Laxity")

    plt.yscale('log')  # important: BFs vary hugely
    plt.xlabel("Life proportion in stringent laws")
    plt.ylabel("Bayes Factor (Design over Non-Design)")
    plt.title("BF for Life vs BF for Laxity\n(Lax always more life-dense)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_bfs()