import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = r"C:\MDM2\dogattackdatabase-USA-fatal - dogattackdatabase-USA-fatal.csv"  # Use raw string for Windows file path


class DataHandler():
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.feature_matrix = pd.DataFrame()  # Initialize feature_matrix

    def clean(self):
        self.feature_matrix = self.feature_matrix.dropna()

    def vectors(self, breed, i):
        # Extract columns from the DataFrame
        age = self.df['Vic age']
        numeric_age = pd.to_numeric(age, errors='coerce')  # Convert to numeric, invalid parsing will be set as NaN
        age = numeric_age.dropna()
        sex = self.df['Vic sex'].replace({'M': 1, 'F': 0})
        sex = sex[[i != 'N' for i in sex]]

        # One-hot encode the breed column

        # Create a feature matrix as a DataFrame
        self.feature_matrix = pd.DataFrame({
            'Vic age': age,
            'Vic sex': sex,
            i:breed
        })

    def breed_encoding(self):
        # One-hot encode the breed column, creating binary columns for each breed
        breed_encoded = pd.get_dummies(self.df['Primary breed'], drop_first=True)  # Set drop_first=True to avoid multicollinearity if needed
        breed_encoded = breed_encoded.astype(int)

        return breed_encoded

    def correlation_matrix(self):
        correlation = self.feature_matrix.corr()

        # Visualize the correlation matrix
        plt.figure(figsize=(12, 8))
        plt.imshow(correlation, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.xticks(ticks=range(len(correlation.columns)), labels=correlation.columns, rotation=90)
        plt.yticks(ticks=range(len(correlation.columns)), labels=correlation.columns)
        plt.title('Correlation Matrix with One-Hot Encoded Breeds')
        plt.show()

    def histogram(self):
        # Get the counts of each unique breed
        breed_counts = self.df['Primary breed'].value_counts()

        # Create a bar chart instead of a histogram
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        breed_counts.plot(kind='bar')  # Use bar plot for categorical data
        plt.xlabel('Breed')
        plt.ylabel('Count')
        plt.title('Count of Each Dog Breed')
        plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
        plt.grid(axis='y')  # Add gridlines for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
        plt.show()


# Example usage
dh = DataHandler(filename)
breeds = dh.breed_encoding()
for i in breeds.columns:
    dh.vectors(breeds[i], i)
    dh.clean()
    dh.correlation_matrix()

dh.histogram()


