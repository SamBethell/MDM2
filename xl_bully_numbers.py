import pandas as pd

data = pd.read_csv("xlbullies_by_postcode.csv")
numbers = [int(i) if i != "0-9" else 4 for i in data["Number of XL Bully applications approved"]]

print(f"Total number of XL Bullies: {sum(numbers)}")
