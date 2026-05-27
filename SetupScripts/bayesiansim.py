import numpy as np
import matplotlib.pyplot as plt


def bayes_factor_laxity(life_lax, nonlife_lax, life_stringent, nonlife_stringent):
    """
    BF for H2 over H1 where:
    H1 = indifference over all worlds
    H2 = indifference over life-worlds only

    Observation = we observe lax laws

    Returns:
        p_lax_h1, p_lax_h2, bf_h2_over_h1
    """
    total_lax = life_lax + nonlife_lax
    total_stringent = life_stringent + nonlife_stringent
    total_worlds = total_lax + total_stringent
    total_life = life_lax + life_stringent

    if total_worlds == 0:
        raise ValueError("Total worlds cannot be 0.")
    if total_life == 0:
        raise ValueError("Total life-worlds cannot be 0.")
    if total_lax == 0:
        raise ValueError("Total lax worlds cannot be 0.")

    p_lax_h1 = total_lax / total_worlds
    p_lax_h2 = life_lax / total_life
    bf_h2_over_h1 = p_lax_h2 / p_lax_h1

    return p_lax_h1, p_lax_h2, bf_h2_over_h1


def plot_bf_vs_stringent_density(total_lax, lax_life_prop, total_stringent, n_points=200):
    """
    Plot BF(H2 over H1) as the life-density of stringent worlds changes.

    Assumption:
    stringent life proportion is always strictly less than lax life proportion.
    """
    if total_lax <= 0 or total_stringent <= 0:
        raise ValueError("Total lax and total stringent worlds must both be > 0.")
    if not (0 < lax_life_prop <= 1):
        raise ValueError("lax_life_prop must be in (0, 1].")

    # Fixed lax counts
    life_lax = lax_life_prop * total_lax
    nonlife_lax = total_lax - life_lax

    # Vary stringent life proportion, but always keep it below lax_life_prop
    stringent_props = np.linspace(0.001, max(0.001, lax_life_prop - 0.001), n_points)
    bfs = []

    for s_prop in stringent_props:
        life_stringent = s_prop * total_stringent
        nonlife_stringent = total_stringent - life_stringent
        _, _, bf = bayes_factor_laxity(
            life_lax, nonlife_lax, life_stringent, nonlife_stringent
        )
        bfs.append(bf)

    plt.figure(figsize=(8, 5))
    plt.plot(stringent_props, bfs)
    plt.axhline(1, linestyle='--')
    plt.xlabel("Proportion of life-worlds within stringent structures")
    plt.ylabel("Bayes factor for H2 over H1 on observing laxity")
    plt.title("How BF changes as stringent life-density changes\n(lax density held higher throughout)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Enter the counts for your actual case:\n")

    life_lax = float(input("Life-worlds under lax laws: "))
    nonlife_lax = float(input("Non-life worlds under lax laws: "))
    life_stringent = float(input("Life-worlds under stringent laws: "))
    nonlife_stringent = float(input("Non-life worlds under stringent laws: "))

    p_lax_h1, p_lax_h2, bf = bayes_factor_laxity(
        life_lax, nonlife_lax, life_stringent, nonlife_stringent
    )

    total_lax = life_lax + nonlife_lax
    total_stringent = life_stringent + nonlife_stringent
    lax_life_prop = life_lax / total_lax

    print("\nResults for your entered values:")
    print(f"P(lax | H1: all worlds)      = {p_lax_h1:.6f}")
    print(f"P(lax | H2: life-worlds only)= {p_lax_h2:.6f}")
    print(f"BF (H2 over H1)              = {bf:.6f}")

    if bf > 1:
        print("Observing laxity favors H2.")
    elif bf < 1:
        print("Observing laxity favors H1.")
    else:
        print("Observing laxity is neutral between H1 and H2.")

    # Plot BF as stringent density varies, while lax stays more life-dense
    plot_bf_vs_stringent_density(
        total_lax=total_lax,
        lax_life_prop=lax_life_prop,
        total_stringent=total_stringent
    )