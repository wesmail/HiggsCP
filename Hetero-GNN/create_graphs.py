import numpy as np
import pandas as pd
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Process the CSV file.')
parser.add_argument('input_process', type=str, help='Name of the Process')
args = parser.parse_args()

# Extract channel name from input file
channel = args.input_process

df1 = pd.read_csv("../kinematics_lowlevel_"+channel+".csv")
df2 = pd.read_csv("../kinematics_highlevel_"+channel+".csv")
df3 = pd.read_csv("../kinematics_polarisation_"+channel+".csv")

df = pd.concat([df1, df2, df3], axis=1, join="inner")

keep = ['# pTl1 ', 'etal1', 'phil1', 'El1', 'pTl2', 'etal2', 'phil2', 'El2',\
        'MET', 'phiMiss', 'pTb1', 'etab1', 'phib1', 'Eb1', 'pTb2', 'etab2',\
        'phib2', 'Eb2', 'pTb3', 'etab3', 'phib3', 'Eb3', 'pTb4', 'etab4',\
        'phib4', 'Eb4', '# pTt1', 'etat1', 'phit1', 'Et1', 'pTt2', 'etat2',\
        'phit2', 'Et2', 'pTh', 'etah', 'phih', 'Eh', 'dphi_tt', 'DeltaPhiLL', 'costhetaS1']

replace = ['pT_lepton1', 'eta_lepton1', 'phi_lepton1', 'E_lepton1', \
           'pT_lepton2', 'eta_lepton2', 'phi_lepton2', 'E_lepton2', \
           'MET', 'phi_MET', \
           'pT_bjet1', 'eta_bjet1', 'phi_bjet1', 'E_bjet1',\
           'pT_bjet2', 'eta_bjet2', 'phi_bjet2', 'E_bjet2',\
           'pT_bjet3', 'eta_bjet3', 'phi_bjet3', 'E_bjet3',\
           'pT_bjet4', 'eta_bjet4', 'phi_bjet4', 'E_bjet4',\
           'pT_top1', 'eta_top1', 'phi_top1', 'E_top1',\
           'pT_top2', 'eta_top2', 'phi_top2', 'E_top2',\
           'pT_higgs', 'eta_higgs', 'phi_higgs', 'E_higgs',\
           'phi_top1_top2', 'DeltaPhiLL', 'costhetaS1']

df = df[keep]
df.columns = replace

# Initialize an empty list to store the feature vectors
feature_vectors = []
eventId = 0

# Loop through each row in the DataFrame to create the feature vector for each event
for index, row in df.iterrows():
    lepton1 = [1, 0, 0, 0, row['pT_lepton1'], row['E_lepton1'], row['eta_lepton1'], row['phi_lepton1'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    lepton2 = [1, 0, 0, 0, row['pT_lepton2'], row['E_lepton2'], row['eta_lepton2'], row['phi_lepton2'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    bjet1   = [0, 1, 0, 0, row['pT_bjet1'], row['E_bjet1'], row['eta_bjet1'], row['phi_bjet1'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    bjet2   = [0, 1, 0, 0, row['pT_bjet2'], row['E_bjet2'], row['eta_bjet2'], row['phi_bjet2'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    bjet3   = [0, 1, 0, 0, row['pT_bjet3'], row['E_bjet3'], row['eta_bjet3'], row['phi_bjet3'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    bjet4   = [0, 1, 0, 0, row['pT_bjet4'], row['E_bjet4'], row['eta_bjet4'], row['phi_bjet4'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    met     = [0, 0, 1, 0, row['MET'], row['MET'], row['phi_MET'], row['phi_MET'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]

    top1    = [0, 0, 0, 1, row['pT_top1'], row['E_top1'], row['eta_top1'], row['phi_top1'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    top2    = [0, 0, 0, 1, row['pT_top2'], row['E_top2'], row['eta_top2'], row['phi_top2'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    higgs   = [0, 0, 0, 0, row['pT_higgs'], row['E_higgs'], row['eta_higgs'], row['phi_higgs'], row['DeltaPhiLL'], row['phi_top1_top2'], eventId]
    
    event_feature_vector = [lepton1, lepton2, bjet1, bjet2, bjet3, bjet4, met, top1, top2, higgs]
    feature_vectors.append(event_feature_vector)
    eventId += 1

# Convert the list of feature vectors into a DataFrame
feature_df = pd.DataFrame(np.vstack(feature_vectors), columns=['I1', 'I2', 'I3', 'It', 'pT', 'E', 'Eta', 'Phi', 'll_angle', 'tt_angle', 'event_id'])

# Save the DataFrame to a CSV file
output_file = f"hetero_graphs_{channel}.csv"
feature_df.to_csv(output_file, index=False)