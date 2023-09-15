import numpy as np
import pandas as pd
import argparse
import os

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Process kinematics CSV file.')
parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
args = parser.parse_args()

# Extract channel name from input file
channel = os.path.basename(args.input_file).split('_')[2].split('.')[0]  # Assumes format is "kinematics_lowlevel_CHANNEL.csv"

# Read the input CSV file
df = pd.read_csv(args.input_file)
#df = df.sample(n=9531) # Maximum number of BACKGROUND events, COMMENT if BINARY classification
print(list(df.columns))

# Initialize an empty list to store the feature vectors
feature_vectors = []
eventId = 0

# Loop through each row in the DataFrame to create the feature vector for each event
for index, row in df.iterrows():
    lepton1 = [1, 0, 0, row['# pTl1 '], row['El1'], row['etal1'], row['phil1'], eventId]
    lepton2 = [1, 0, 0, row['pTl2'], row['El2'], row['etal2'], row['phil2'], eventId]
    bjet1 = [0, 1, 0, row['pTb1'], row['Eb1'], row['etab1'], row['phib1'], eventId]
    bjet2 = [0, 1, 0, row['pTb2'], row['Eb2'], row['etab2'], row['phib2'], eventId]
    bjet3 = [0, 1, 0, row['pTb3'], row['Eb3'], row['etab3'], row['phib3'], eventId]
    bjet4 = [0, 1, 0, row['pTb4'], row['Eb4'], row['etab4'], row['phib4'], eventId]
    met = [0, 0, 1, row['MET'], row['MET'], row['phiMiss'], row['phiMiss'], eventId]
    
    event_feature_vector = [lepton1, lepton2, bjet1, bjet2, bjet3, bjet4, met]
    feature_vectors.append(event_feature_vector)
    eventId += 1

# Convert the list of feature vectors into a DataFrame
feature_df = pd.DataFrame(np.vstack(feature_vectors), columns=['I1', 'I2', 'I3', 'pT', 'E', 'Eta', 'Phi', 'event_id'])

# Save the DataFrame to a CSV file
output_file = f"graphs_{channel}.csv"
feature_df.to_csv(output_file, index=False)
