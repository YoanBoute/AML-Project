# As the number of experiments is high, the script is splitted in two parts to divide computation time

# Baseline experiment for all target domains
source ./launch_scripts/baseline.sh cartoon
source ./launch_scripts/baseline.sh photo
source ./launch_scripts/baseline.sh sketch


# Random experiment, which adds an ASM after each layer of the network, and tests the performance on target domains with random maps containing between 20% and 100% of 1
source ./launch_scripts/random_with_ratios.sh cartoon 1.0
rm ./record/random/cartoon/last.pth
source ./launch_scripts/random_with_ratios.sh cartoon 0.8
rm ./record/random/cartoon/last.pth
source ./launch_scripts/random_with_ratios.sh cartoon 0.6
rm ./record/random/cartoon/last.pth
source ./launch_scripts/random_with_ratios.sh cartoon 0.4
rm ./record/random/cartoon/last.pth
source ./launch_scripts/random_with_ratios.sh cartoon 0.2
rm ./record/random/cartoon/last.pth

source ./launch_scripts/random_with_ratios.sh photo 1.0
rm ./record/random/photo/last.pth
source ./launch_scripts/random_with_ratios.sh photo 0.8
rm ./record/random/photo/last.pth
source ./launch_scripts/random_with_ratios.sh photo 0.6
rm ./record/random/photo/last.pth
source ./launch_scripts/random_with_ratios.sh photo 0.4
rm ./record/random/photo/last.pth
source ./launch_scripts/random_with_ratios.sh photo 0.2
rm ./record/random/photo/last.pth

source ./launch_scripts/random_with_ratios.sh sketch 1.0
rm ./record/random/sketch/last.pth
source ./launch_scripts/random_with_ratios.sh sketch 0.8
rm ./record/random/sketch/last.pth
source ./launch_scripts/random_with_ratios.sh sketch 0.6
rm ./record/random/sketch/last.pth
source ./launch_scripts/random_with_ratios.sh sketch 0.4
rm ./record/random/sketch/last.pth
source ./launch_scripts/random_with_ratios.sh sketch 0.2
rm ./record/random/sketch/last.pth

read -n1 -r -p "Press any key to continue..."