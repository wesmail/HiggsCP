import sys
import h5py
import argparse
import numpy as np
from tqdm.auto import tqdm


def pad(a, features=8):
    """
    Pads a NumPy array with zeros to ensure a shape of (features,)

    Args:
        a: The input NumPy array.

    Returns:
        A new NumPy array with shape (features,) padded with zeros.
    """
    padding = (0, features - a.shape[0]) if a.shape[0] < features else (0, 0)
    return np.pad(a, padding, mode="constant", constant_values=0)


def load_events(root_dir, channel="S", angle=""):
    angle = "_" + angle if angle else ""
    # tau constitutents
    sig_tau1_constitutents = np.load(
        root_dir + "df_" + channel + "_C1.pkl", allow_pickle=True
    )
    sig_tau2_constitutents = np.load(
        root_dir + "df_" + channel + "_C2.pkl", allow_pickle=True
    )

    # reconstructed tau
    sig_tau1_reco = np.load(root_dir + "df_" + channel + "_ta1.pkl", allow_pickle=True)
    sig_tau2_reco = np.load(root_dir + "df_" + channel + "_ta2.pkl", allow_pickle=True)

    # the system of two reconstructed tau
    sig_higgs = np.load(root_dir + "df_" + channel + "_tata.pkl", allow_pickle=True)

    # the angular distribution between the two taus
    sig_angluar_distribution = np.load(
        root_dir + "df_" + channel + angle + "_theta.pkl", allow_pickle=True
    )

    events = []

    n_events = 10000  # sig_angluar_distribution.shape[0]
    for item in tqdm(
        range(n_events), total=n_events, desc="Processing " + channel + " Events "
    ):
        row1 = sig_tau1_constitutents.iloc[item]
        row2 = sig_tau2_constitutents.iloc[item]
        row3 = sig_tau1_reco.iloc[item]
        row4 = sig_tau2_reco.iloc[item]
        row5 = sig_higgs.iloc[item]
        row6 = sig_angluar_distribution.iloc[item]

        a1, a2 = [], []
        for j in range(8):
            a1.append(row1.iloc[j])
            a2.append(row2.iloc[j])

        a1 = np.vstack(a1).T
        a2 = np.vstack(a2).T

        a3 = pad(row3.to_numpy())
        a4 = pad(row4.to_numpy())
        a5 = pad(row5.to_numpy())
        a6 = pad(row6.to_numpy())

        events.append(np.vstack([a1, a2, a3, a4, a5, a6]))

    events = np.stack(events)
    return events


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir",
        type=str,
        required=False,
        help="input directory",
        default="./Files/",
    )

    parser.add_argument(
        "-angle",
        type=str,
        required=False,
        help="input angle",
        default="0",
    )

    args = parser.parse_args()

    sig = load_events(root_dir=args.dir, channel="S", angle=args.angle)
    bkg = load_events(root_dir=args.dir, channel="B", angle=args.angle)
    sig_labels = np.ones(sig.shape[0])
    bkg_labels = np.zeros(bkg.shape[0])

    events = np.concatenate([sig, bkg])
    labels = np.concatenate([sig_labels, bkg_labels])

    with h5py.File(args.dir + "data_" + args.angle + ".hdf5", "w") as h5file:
        # Store each array as a separate dataset
        h5file.create_dataset("data", data=events)
        h5file.create_dataset("labels", data=labels)


if __name__ == "__main__":
    main(sys.argv[1:])
