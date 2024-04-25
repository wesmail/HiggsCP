#!/bin/bash

# Define angles to train on
ANGLES=(0 15 30 45 60 90)

# Loop through each angle
for ANGLE in "${ANGLES[@]}"; do
  echo "--------------------------------------------------START---------------------------------------------------"
  # Call run.sh with train mode and current angle
  ./run.sh -m train -a "$ANGLE"

  # Optional: Add a separator between training runs
  echo "--------------------------------------------------END----------------------------------------------------"
done

echo "Training completed for all angles!"
