import csv

def generate_condition_csv(
    n_conditions,
    output_file="conditions.csv",
    base_url="https://run.pavlovia.org/montesinos7/test/?condition=",
    pad=3
):
    """
    Generate a CSV file with Pavlovia condition URLs.

    Parameters
    ----------
    n_conditions : int
        Number of conditions to generate.
    output_file : str
        Name of the CSV file to save.
    base_url : str
        URL prefix before the condition number.
    pad : int
        Zero-padding length for condition numbers (e.g., 3 → '001').
    """

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        for i in range(1, n_conditions + 1):
            cond_str = str(i).zfill(pad)
            url = f"{base_url}{cond_str}"
            writer.writerow([url])

    print(f"Saved {n_conditions} conditions to {output_file}")

'''

generate_condition_csv(
    n_conditions=30,
    output_file="/Users/sm6511/Desktop/Prediction-Accomodation-Exp/ConditionFiles-Prolific/conditions_accomodate.csv",
    base_url="https://run.pavlovia.org/montesinos7/explain2?condition="
)

generate_condition_csv(
    n_conditions=30,
    output_file="/Users/sm6511/Desktop/Prediction-Accomodation-Exp/ConditionFiles-Prolific/conditions_predict.csv",
    base_url="https://run.pavlovia.org/montesinos7/test?condition="
)
'''
def generate_condition_combined_csv(
    n_conditions,
    output_file="conditions.csv",
    base_url="https://run.pavlovia.org/montesinos7/test/?condition=",
    base_url2="https://run.pavlovia.org/montesinos7/explain2?condition=",
    pad=3
):

    """
    Generate a CSV file with Pavlovia condition URLs.

    Parameters
    ----------
    n_conditions : int
        Number of conditions to generate.
    output_file : str
        Name of the CSV file to save.
    base_url : str
        URL prefix before the condition number.
    pad : int
        Zero-padding length for condition numbers (e.g., 3 → '001').
    """

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        for i in range(1, n_conditions + 1):
            cond_str = str(i).zfill(pad)
            url1 = f"{base_url}{cond_str}"
            url2 = f"{base_url2}{cond_str}"
            writer.writerow([url1])
            writer.writerow([url2])

    print(f"Saved {n_conditions} conditions to {output_file}")

generate_condition_combined_csv(
    n_conditions=6,
    output_file="/Users/sm6511/Desktop/Prediction-Accomodation-Exp/ConditionFiles-Prolific/conditions_combined.csv",
    base_url="https://run.pavlovia.org/montesinos7/test?condition=",
    base_url2="https://run.pavlovia.org/montesinos7/explain2?condition="
)