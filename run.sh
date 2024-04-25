#!/bin/bash

# Function to display help message
usage() {
  echo "Usage: $0 [OPTION]..."
  echo ""
  echo "Script to process h -> ttbar events, train a model, or test on a trained model."
  echo ""
  echo "Options:"
  echo "  -h, --help          Display this help message."
  echo "  -d, --dir DIR       Input directory containing event data (default: ./files/)."
  echo "  -a, --angle ANGLE    Input angle (default: 0)."
  echo "  -m, --mode MODE     (required) Mode (train or test)."
  echo "  -p, --ckpt_path PATH  Path to the checkpoint file for testing (required with --mode=test)."
  echo "  -f, --h5_file FILE   Path to the HDF5 file for testing (required with --mode=test)."
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    -d|--dir)
      shift
      # Ensure directory ends with '/'
      INPUT_DIR="${1%*/}"  # Remove trailing slash if present
      if [[ ! "$INPUT_DIR" =~ "/" ]]; then
        INPUT_DIR="$INPUT_DIR/"
      fi
      ;;
    -a|--angle)
      shift
      ANGLE=$1
      ;;
    -m|--mode)
      shift
      MODE="$1"
      ;;
    -p|--ckpt_path)
      shift
      CKPT_PATH="$1"
      ;;
    -f|--h5_file)
      shift
      H5_FILE="$1"
      ;;
    *)
      echo "Error: Unknown option '$1'"
      usage
      exit 1
      ;;
  esac
  shift
done

# Check for required arguments
if [[ -z "$MODE" ]]; then
  echo "Error: --mode argument is required (train or test)."
  usage
  exit 1
fi

if [[ "$MODE" == "test" ]]; then
  if [[ -z "$CKPT_PATH" || -z "$H5_FILE" ]]; then
    echo "Error: --ckpt_path and --h5_file arguments are required for testing mode."
    usage
    exit 1
  fi
fi

# Set default values if options not provided
INPUT_DIR=${INPUT_DIR:-"./files/"}
ANGLE=${ANGLE:-"0"}

if [[ "$MODE" == "train" ]]; then
  echo "Creating Graphs for the angle $ANGLE"

  # Run data processing script
  ipython ./data/create_graphs.py -- -dir "$INPUT_DIR" -angle "$ANGLE"

  # Get the HDF5 file name
  HDF5_FILE="$INPUT_DIR""data_${ANGLE}.hdf5"
  OUT_DIR="h2tt_angle_"$ANGLE"_results/"
  # Training script execution
  echo "Training model with $HDF5_FILE"
  ipython main.py -- fit --config configs/train.yaml --data.h5_file "$HDF5_FILE" --trainer.logger.name "$OUT_DIR"

elif [[ "$MODE" == "test" ]]; then
  # Testing mode checks
  if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: --ckpt_path argument required for testing mode."
    exit 1
  fi

  # Testing script execution
  echo "Testing model with checkpoint: $CKPT_PATH"
  ipython main.py -- test --config configs/train.yaml --data.h5_file "$H5_FILE" --ckpt_path "$CKPT_PATH"

else
  echo "Error: Invalid mode '$MODE'. Please use train or test."
  exit 1
fi
