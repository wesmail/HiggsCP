import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input dataframe by removing outlier events.

    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe.

    Returns
    -------
    pd.DataFrame
        The input dataframe with outlier events removed and the energy of the particles set to zero if it is above a certain threshold.

    """
    # Clean the data
    df_filter = df[(df["p_top.M()"] < 173 + 40) & (df["p_top.M()"] > 173 - 40)]
    df_filter = df_filter[
        (df_filter["p_v1.E()"] < 1000) & (df_filter["p_v2.E()"] < 1000)
    ]
    return df_filter


lowlevel_clms = [
    "p_l.first.Pt()",
    "p_l.first.Eta()",
    "p_l.first.Phi()",
    "p_l.first.E()",
    "p_b1.Pt()",
    "p_b1.Eta()",
    "p_b1.Phi()",
    "p_b1.E()",
    "p_v1.Pt()",
    "p_v1.Eta()",
    "p_v1.Phi()",
    "p_v1.E()",
    "p_l.second.Pt()",
    "p_l.second.Eta()",
    "p_l.second.Phi()",
    "p_l.second.E()",
    "p_b2.Pt()",
    "p_b2.Eta()",
    "p_b2.Phi()",
    "p_b2.E()",
    "p_v2.Pt()",
    "p_v2.Eta()",
    "p_v2.Phi()",
    "p_v2.E()",
    "pb_fromHiggs_1.Pt()",
    "pb_fromHiggs_1.Eta()",
    "pb_fromHiggs_1.Phi()",
    "pb_fromHiggs_1.E()",
    "pb_fromHiggs_2.Pt()",
    "pb_fromHiggs_2.Eta()",
    "pb_fromHiggs_2.Phi()",
    "pb_fromHiggs_2.E()",
    "process",
]

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Process kinematics CSV file.")
parser.add_argument(
    "--input_file",
    type=str,
    help="Path to the input CSV file.",
    required=False,
    default="MLP-Data.csv",
)
parser.add_argument(
    "--file_dir",
    type=str,
    help="Directory to where the file is.",
    required=False,
    default="/mnt/d/waleed/CP-Higgs/Reco/Files/",
)
args = parser.parse_args()

# Read the input CSV file
df = pd.read_csv(args.file_dir + args.input_file)
df = clean(df=df)
channel = "qcd"
eventId = 0
for label in [0, 1, 2]:
    print(f"Processing {channel} {label} events")
    clas = df[df.label == label]
    clas = clas[lowlevel_clms]
    if label == 1:
        channel = "cpodd"
    elif label == 2:
        channel = "cpeven"

    # Initialize an empty list to store the feature vectors
    feature_vectors = []
    for row in tqdm(
        clas.to_dict("records"), total=len(clas), desc="Processing Events "
    ):
        lepton1 = [
            1,
            0,
            0,
            row["p_l.first.Pt()"],
            row["p_l.first.E()"],
            row["p_l.first.Eta()"],
            row["p_l.first.Phi()"],
        ]

        lepton2 = [
            1,
            0,
            0,
            row["p_l.second.Pt()"],
            row["p_l.second.E()"],
            row["p_l.second.Eta()"],
            row["p_l.second.Phi()"],
        ]

        bjet1 = [
            0,
            1,
            0,
            row["p_b1.Pt()"],
            row["p_b1.E()"],
            row["p_b1.Eta()"],
            row["p_b1.Phi()"],
        ]

        bjet2 = [
            0,
            1,
            0,
            row["p_b2.Pt()"],
            row["p_b2.E()"],
            row["p_b2.Eta()"],
            row["p_b2.Phi()"],
        ]

        bjet3 = [
            0,
            1,
            0,
            row["pb_fromHiggs_1.Pt()"],
            row["pb_fromHiggs_1.E()"],
            row["pb_fromHiggs_1.Eta()"],
            row["pb_fromHiggs_1.Phi()"],
        ]

        bjet4 = [
            0,
            1,
            0,
            row["pb_fromHiggs_2.Pt()"],
            row["pb_fromHiggs_2.E()"],
            row["pb_fromHiggs_2.Eta()"],
            row["pb_fromHiggs_2.Phi()"],
        ]

        met1 = [
            0,
            0,
            1,
            row["p_v1.Pt()"],
            row["p_v1.E()"],
            row["p_v1.Eta()"],
            row["p_v1.Phi()"],
        ]
        met2 = [
            0,
            0,
            1,
            row["p_v2.Pt()"],
            row["p_v2.E()"],
            row["p_v2.Eta()"],
            row["p_v2.Phi()"],
        ]

        event_feature_vector = [
            lepton1,
            lepton2,
            bjet1,
            bjet2,
            bjet3,
            bjet4,
            met1,
            met2,
        ]
        feature_vectors.append(event_feature_vector)
        eventId += 1

    features = np.stack(feature_vectors)
    features = features.reshape(features.shape[0], -1)
    # Convert the list of feature vectors into a DataFrame
    feature_df = pd.DataFrame(
        features,
        columns=[
            "Lepton1-I1",
            "Lepton1-I2",
            "Lepton1-I3",
            "Lepton1-pT",
            "Lepton1-E",
            "Lepton1-Eta",
            "Lepton1-Phi",
            "Lepton2-I1",
            "Lepton2-I2",
            "Lepton2-I3",
            "Lepton2-pT",
            "Lepton2-E",
            "Lepton2-Eta",
            "Lepton2-Phi",
            "Bjet1-I1",
            "Bjet1-I2",
            "Bjet1-I3",
            "Bjet1-pT",
            "Bjet1-E",
            "Bjet1-Eta",
            "Bjet1-Phi",
            "Bjet2-I1",
            "Bjet2-I2",
            "Bjet2-I3",
            "Bjet2-pT",
            "Bjet2-E",
            "Bjet2-Eta",
            "Bjet2-Phi",
            "Bjet3-I1",
            "Bjet3-I2",
            "Bjet3-I3",
            "Bjet3-pT",
            "Bjet3-E",
            "Bjet3-Eta",
            "Bjet3-Phi",
            "Bjet4-I1",
            "Bjet4-I2",
            "Bjet4-I3",
            "Bjet4-pT",
            "Bjet4-E",
            "Bjet4-Eta",
            "Bjet4-Phi",
            "MET1-I1",
            "MET1-I2",
            "MET1-I3",
            "MET1-pT",
            "MET1-E",
            "MET1-Eta",
            "MET1-Phi",
            "MET2-I1",
            "MET2-I2",
            "MET2-I3",
            "MET2-pT",
            "MET2-E",
            "MET2-Eta",
            "MET2-Phi",
        ],
    )

    # Save the DataFrame to a CSV file
    output_file = f"graphs_{channel}.csv"
    print(f"saving {channel} {label} events to {output_file}")
    feature_df.to_csv(output_file, index=False)
