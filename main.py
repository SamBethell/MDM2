import pandas as pd
import matplotlib.pyplot as plt


def dog_attack_rise_per_state():
    dog_attack_data = [
        {"State": "California", "2023": 727, "2022": 675},
        {"State": "Texas", "2023": 411, "2022": 404},
        {"State": "Ohio", "2023": 359, "2022": 311},
        {"State": "Pennsylvania", "2023": 334, "2022": 313},
        {"State": "Illinois", "2023": 316, "2022": 245},
        {"State": "New York", "2023": 296, "2022": 321},
        {"State": "Florida", "2023": 193, "2022": 220},
        {"State": "North Carolina", "2023": 185, "2022": 146},
        {"State": "Michigan", "2023": 183, "2022": 206},
        {"State": "Missouri", "2023": 180, "2022": 166}]

    df_dog_attacks = pd.DataFrame(dog_attack_data)

    df_dog_attacks['Rise in Attacks'] = df_dog_attacks['2023'] - df_dog_attacks['2022']

    # Set the figure size for better readability
    plt.figure(figsize=(10, 6))

    # Bar plot for the rise in attacks
    plt.bar(df_dog_attacks['State'], df_dog_attacks['Rise in Attacks'], color='skyblue')

    # Add titles and labels
    plt.title('Rise in Dog Attacks by State (2023 vs. 2022)', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Rise in Attacks', fontsize=12)

    # Display the rise in attacks on the bars
    for index, value in enumerate(df_dog_attacks['Rise in Attacks']):
        plt.text(index, value, str(value), ha='center', va='bottom')

    # Show the plot
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

dog_attack_rise_per_state()
