import os
import pandas as pd
from utils import imodels_classification_datasets, load_imodels_data

# Create the 'data' folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load and save each dataset
for dataset_name in imodels_classification_datasets:
    dataset = load_imodels_data(dataset_name)
    df = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
    df['target'] = dataset["target"]
    
    # Save the dataset as a CSV file
    # Create a folder with the dataset name
    dataset_folder = os.path.join('../data', "imodels_"+dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    # Save the dataset as a CSV file in the folder
    csv_path = os.path.join(dataset_folder, f'imodels_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Dataset {dataset_name} saved to {csv_path}')
    