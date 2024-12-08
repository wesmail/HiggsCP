import sys
import h5py
import argparse
import numpy as np
from tqdm.auto import tqdm


def sample(arr, frac):
    """Sample a fraction of events from a 3D numpy array along the first dimension."""
    if len(arr.shape) != 3:
        raise ValueError("This function only supports 3D numpy arrays.")

    n = int(arr.shape[0] * frac)  # Calculate the number of events to sample
    shuffled_indices = np.random.permutation(
        arr.shape[0]
    )  # Shuffle indices of the first dimension
    sampled_indices = shuffled_indices[:n]  # Take the first n shuffled indices
    return arr[sampled_indices, :, :]  # Sample along the first dimension


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


def load_events(root_dir, channel="S", angle="", num_events=0):
    # tau constitutents
    tau_constitutents = np.load(
        root_dir + "constitutes_" + channel + "_" + angle + ".pkl", allow_pickle=True
    )

    # tau jets
    tau_jets = np.load(
        root_dir + "observables_" + channel + "_" + angle + ".npz", allow_pickle=False
    )["arr_0"]

    num_events = tau_constitutents.shape[0] if num_events == 0 else num_events
    events = []
    for item in tqdm(
        range(num_events), total=num_events, desc="Processing " + channel + " Events "
    ):
        row1 = tau_constitutents.iloc[item]
        row2 = tau_jets[item]

        a1, a2 = [], []
        for j in range(0, 8, 1):
            a1.append(row1.iloc[j])
        for j in range(8, 16, 1):
            a2.append(row1.iloc[j])

        a1 = np.vstack(a1).T  # Tau1 const. 0-2
        a2 = np.vstack(a2).T  # Tau2 const. 3-5

        a3 = pad(a=row2[6:11])  # Tau1 jet. 6
        a4 = pad(a=row2[11:16])  # Tau2 jet. 7
        a5 = pad(a=row2[5:6])  # MET 8
        a6 = pad(a=row2[0:5])  # Higgs 9
        a7 = pad(a=row2[16:17])  # Angle 10

        events.append(np.vstack([a1, a2, a3, a4, a5, a6, a7]))

    events = np.stack(events)

    return events


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dir",
        type=str,
        required=False,
        help="input directory",
        default="./files/",
    )

    parser.add_argument(
        "-angle",
        type=str,
        required=False,
        help="input angle",
        default="00p",
    )

    parser.add_argument(
        "-num_events",
        type=int,
        required=False,
        help="number of events",
        default=0,
    )

    args = parser.parse_args()

    sig = load_events(
        root_dir=args.dir, channel="sig", angle=args.angle, num_events=args.num_events
    )
    bkg = load_events(
        root_dir=args.dir, channel="bkg", angle=args.angle, num_events=args.num_events
    )
    # Sample some events from the background
    # bkg = sample(arr=bkg, frac=0.5)
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
