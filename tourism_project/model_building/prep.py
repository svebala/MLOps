
# for data manipulation
import pandas as pd

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/BalaSVenkat/tourism-package-dataset/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")
print(f"Initial Dataset Shape: {tourism_dataset.shape}")

# Remove unwanted columns
unwanted_cols = ["Unnamed: 0", "CustomerID"]
tourism_dataset = tourism_dataset.drop(columns=unwanted_cols,errors="ignore")
print(f"Dataset shape after dropping columns: {tourism_dataset.shape}")

# Fix Gender category inconsistency
tourism_dataset["Gender"] = tourism_dataset["Gender"].replace({"Fe Male": "Female"})

# Fix MaritalStatus category inconsistency
tourism_dataset["MaritalStatus"] = (tourism_dataset["MaritalStatus"].astype(str).str.strip().str.title())
tourism_dataset["MaritalStatus"] = tourism_dataset["MaritalStatus"].replace({"Unmarried": "Single"})

# Feature Engineering - Age Group
age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
age_labels = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64"
]
tourism_dataset["AgeGroup"] = pd.cut(tourism_dataset["Age"],bins=age_bins,labels=age_labels,right=False)

# Drop original Age column
tourism_dataset.drop(columns=["Age"], inplace=True)

# Handle Missing Values
numeric_cols = tourism_dataset.select_dtypes(include=["int64", "float64"]).columns

categorical_cols = tourism_dataset.select_dtypes(include=["object", "category"]).columns

# Numerical columns -> Median Imputation
for col in numeric_cols:
    tourism_dataset[col] = tourism_dataset[col].fillna(tourism_dataset[col].median())

# Categorical columns -> Mode Imputation
for col in categorical_cols:
    tourism_dataset[col] = tourism_dataset[col].fillna(tourism_dataset[col].mode()[0])

# Remove duplicate records
tourism_dataset = tourism_dataset.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after duplicate removals: {tourism_dataset.shape}")

# Save Cleaned Dataset
tourism_dataset.to_csv(cleaned_tourism.csv,index=False)

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'CityTier',                 # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'NumberOfPersonVisiting',   # Total number of people accompanying the customer on the trip
    'PreferredPropertyStar',    # Preferred hotel rating by the customer
    'NumberOfTrips',            # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome',            # Gross monthly income of the customer
    'PitchSatisfactionScore',   # Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups',        # Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch'           # Duration of the sales pitch delivered to the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',    # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation',       # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',           # Gender of the customer (Male, Female)
    'MaritalStatus',    # Marital status of the customer (Single, Married, Divorced)
    'Designation',      # Customer's designation in their current organization
    'ProductPitched',   # The type of product pitched to the customer
    "AgeGroup",         # Age group of customer
    'Passport',         # Whether the customer holds a valid passport (0: No, 1: Yes)
    'OwnCar'            # Whether the customer owns a car (0: No, 1: Yes)
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.20,    # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # preserves the same class distribution in both train and test datasets.
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["cleaned_tourism.csv","Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="BalaSVenkat/tourism-package-dataset",
        repo_type="dataset",
    )
