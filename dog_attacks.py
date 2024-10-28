import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename = "C:/MDM2/DOHMH_Dog_Bite_Data.csv"


class DataHandler():
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(filename)

    def clean(self):
        # Remove rows with NaN in Age or Gender
        self.df = self.df.dropna(subset=['Age', 'Gender'])

        # Ensure 'Age' is a string type before regex operation
        self.df['Age'] = self.df['Age'].astype(str)

        # Keep only rows where 'Age' is formatted as xY
        self.df = self.df[self.df['Age'].str.contains(r'^\d+Y?$')]

        # Remove 'Y' from 'Age' and convert to integer
        self.df['Age'] = self.df['Age'].str.replace('Y', '', regex=False).astype(int)

        # Filter out rows where Borough is "Other"
        self.df = self.df[self.df['Borough'] != 'Other']

        # Remove rows with unknown gender
        self.df = self.df[self.df['Gender'] != 'U']

        # Convert Gender to binary values
        self.df.loc[self.df['Gender'] == 'M', 'Gender'] = 1
        self.df.loc[self.df['Gender'] == 'F', 'Gender'] = 0

        # Convert 'SpayNeuter' to integer (assuming True/False)
        self.df['SpayNeuter'] = self.df['SpayNeuter'].astype(int)

    def borough(self):
        unique = self.df['Borough'].unique()
        counts = self.df['Borough'].value_counts()
        density = [38634, 34000, 74870.7, 7588, 21178]
        density_dict = dict(zip(unique, density))
        return counts, density_dict

    def breed_encoding(self):
        # One-hot encode the breed column
        breed_encoded = pd.get_dummies(self.df['Breed'], prefix='Breed')
        return breed_encoded

    def select_random_breeds(self, num_breeds=4):
        # Randomly select unique breeds from the dataset
        unique_breeds = self.df['Breed'].unique()
        selected_breeds = np.random.choice(unique_breeds, num_breeds, replace=False)
        return selected_breeds

    def filter_breeds(self, selected_breeds):
        # Filter the dataframe to include only the selected breeds
        self.df = self.df[self.df['Breed'].isin(selected_breeds)]

    def vectors(self):
        # Prepare feature matrix and labels
        age = self.df['Age']
        gender = self.df['Gender']
        sprayneuter = self.df['SpayNeuter']
        counts, density_dict = self.borough()

        # Assign density to the DataFrame by mapping the borough names to their densities
        self.df['density'] = self.df['Borough'].map(density_dict)
        density = self.df['density']
        return age, gender, sprayneuter, density

    def age_bite_histogram(self, number):
        unique_breeds = self.df['Breed'].unique()
        breeds = np.random.choice(unique_breeds, number)  # Select random breeds
        for breed in breeds:
                # Filter DataFrame for the selected breed and extract the 'Age' column
                breed_ages = self.df[self.df['Breed'] == breed]['Age']
                plt.hist(breed_ages, bins=10, alpha=0.7)
                plt.title(f"Age Distribution for {breed}")
                plt.xlabel("Age")
                plt.ylabel("Frequency")
                plt.show()

    def correlation(self, age, gender, sprayneuter, density, breed_encoded):
        # Create a DataFrame from the features and the breed-encoded data
        feature_df = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'SpayNeuter': sprayneuter,
            'density': density
        })

        # Add breed one-hot encoded columns
        feature_df = pd.concat([feature_df, breed_encoded], axis=1)

        # Calculate the correlation matrix
        correlation = feature_df.corr()

        # Visualize the correlation matrix
        plt.figure(figsize=(12, 8))
        plt.imshow(correlation, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(ticks=range(len(feature_df.columns)), labels=feature_df.columns, rotation=90)
        plt.yticks(ticks=range(len(feature_df.columns)), labels=feature_df.columns)
        plt.title('Correlation Matrix Including Randomly Selected Breeds')
        plt.show()


# Initialize DataHandler with the dataset
dh = DataHandler(filename)

# Clean the data
dh.clean()

# Randomly select four breeds
selected_breeds = dh.select_random_breeds(num_breeds=4)

# Filter the dataset to include only the selected breeds
dh.filter_breeds(selected_breeds)

# Generate the vector features
age, gender, sprayneuter, density = dh.vectors()

# Encode the breeds
breed_encoded = dh.breed_encoding()

# Generate the correlation matrix with the selected breeds
dh.correlation(age, gender, sprayneuter, density, breed_encoded)

#dh.age_bite_histogram(5)













