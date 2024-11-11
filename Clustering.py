import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


filename_1 = "DOHMH_Dog_Bite_Data.csv" # fatal attacks
filename_2 = "dogattackdatabase-USA-fatal - dogattackdatabase-USA-fatal.csv" # non-fatal attacks

class DataHandler():
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.filename = filename

    def vectors_non_fatal(self):
        # Drop rows only where 'Vic age' and 'Vic sex' are NaN
        self.df = self.df.dropna(subset=['Vic age', 'Vic sex'])
        scaler = MinMaxScaler()

        # Convert age to numeric, setting errors to NaN and then dropping NaN values
        numeric_age = pd.to_numeric(self.df['Vic age'], errors='coerce')
        self.df = self.df[~numeric_age.isna()]  # Keep rows with valid ages
        age = numeric_age.dropna()

        # Convert 'Vic sex' to binary (M = 1, F = 0) and drop any unknowns ('N')
        sex = self.df['Vic sex'].replace({'M': 1, 'F': 0})
        sex = sex[sex != 'N']
        breed = self.df['Primary breed']
        breeds = pd.get_dummies(breed)
        enc_breeds = breeds.astype(int)
        # Align indexes after filtering
        age = age.loc[sex.index]
        # Create feature matrix
        select_breeds = [breeds['Unknown'], breeds['German Shepherd'], breeds['Rottweiler'], breeds['Pit bull']]
        # Create feature matrix
        self.feature_matrix = pd.DataFrame({
            'Age': age,
            'Gender': sex,
            'unknown':select_breeds[0],
            'german shepherd': select_breeds[1],
            'rottweiler': select_breeds[2],
            'pit bull': select_breeds[3]
        })
        self.label = pd.DataFrame(np.zeros((len(self.feature_matrix), 1)))

    def vectors_fatal(self):
        # Remove rows with NaN in Age or Gender, and unknown Gender ('U')
        self.df = self.df.dropna(subset=['Age', 'Gender'])
        self.df = self.df[self.df['Gender'] != 'U']

        # Ensure 'Age' is a string type before regex operation
        age = self.df['Age'].astype(str)

        # Keep only rows where 'Age' is formatted as xY
        age = age[age.str.contains(r'^\d+Y?$')]

        # Remove 'Y' from 'Age' and convert to integer
        age = age.str.replace('Y', '', regex=False).astype(int)

        # Convert Gender to binary (M = 1, F = 0)
        sex = self.df['Gender'].replace({'M': 1, 'F': 0})
        breeds = pd.get_dummies(self.df["Breed"])
        enc_breeds = breeds.astype(int)
        select_breeds = [enc_breeds['UNKNOWN'], enc_breeds['German Shepherd'], enc_breeds['Rottweiler'], enc_breeds['Pit Bull']]
        # Create feature matrix
        self.feature_matrix = pd.DataFrame({
            'Age': age,
            'Gender': sex,
            'unknown':select_breeds[0],
            'german shepherd': select_breeds[1],
            'rottweiler': select_breeds[2],
            'pit bull': select_breeds[3]
        })
        self.label = pd.DataFrame(np.ones((len(self.feature_matrix), 1)))


class KMeansClustering():
    def __init__(self, fm1, fm2, l1, l2):
        self.fm1 = fm1
        self.fm2 = fm2
        self.l1 = l1
        self.l2 = l2

    def merge_dataframes(self):
        # Concatenate DataFrames along rows (axis=0), since these represent different records
        self.df = pd.concat([self.fm1, self.fm2], axis=0).reset_index(drop=True)
        self.l = pd.concat([self.l1, self.l2], axis=0).reset_index(drop=True)
        # Drop rows with NaN values in the feature matrix
        mask = self.df.notna().all(axis=1)  # Create a mask for rows without NaN
        self.df = self.df[mask]
        self.l = self.l[mask]  # Apply the same mask to labels


    def test_train_split(self):
        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.pc, self.l, test_size=0.2, random_state=42
        )

    def pca(self):
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(self.df.to_numpy())
        plt.figure(figsize=(8, 6))
        plt.title(f'PCA Visualization of Dog attacks')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        c = self.l.to_numpy().flatten()
        self.pc = x_pca
        self.l = c
        scatter = plt.scatter(
            x_pca[:, 0], x_pca[:, 1], c=c, cmap='coolwarm', edgecolor='k'
        )
        plt.colorbar(scatter)
        plt.show()


    def perform_kmeans(self, n_clusters):
        # Fit KMeans model on training data
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(self.X_train)

        # Predict clusters for both training and test sets
        self.Y_train_pred = self.model.predict(self.X_train)
        self.Y_test_pred = self.model.predict(self.X_test)
        train_score = silhouette_score(self.X_train, self.Y_train_pred)
        test_score = silhouette_score(self.X_test, self.Y_test_pred)
        centroids = self.model.cluster_centers_
        print(centroids)
        print(f'Train Silhouette Score: {train_score:.3f}')
        print(f'Test Silhouette Score: {test_score:.3f}')
        print(self.Y_test_pred)
        plt.scatter(self.pc[:, 0], self.pc[:, 1], c=self.l, cmap='coolwarm', edgecolor='k')
        plt.scatter(centroids[0][0], centroids[0][1])
        plt.scatter(centroids[1][0], centroids[1][1])
        plt.show()


# Load and process data
dh1 = DataHandler(filename_1)
dh1.vectors_fatal()
df1, l1 = dh1.feature_matrix, dh1.label

dh2 = DataHandler(filename_2)
dh2.vectors_non_fatal()
df2, l2 = dh2.feature_matrix, dh2.label

# Apply KMeans and PCA
kmeans = KMeansClustering(df1, df2, l1, l2)
kmeans.merge_dataframes()
kmeans.pca()
kmeans.test_train_split()
kmeans.perform_kmeans(2)
