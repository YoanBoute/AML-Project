# set K for all the experiments
K=5

# Random experiment, which adds an ASM after each layer of the network, and tests the performance on target domains with random maps containing between 20% and 100% of 1
source ./launch_scripts/random_top_k.sh cartoon 1.0 $K
rm ./record/random_top_k/cartoon/last.pth
source ./launch_scripts/random_top_k.sh cartoon 0.8 $K
rm ./record/random_top_k/cartoon/last.pth
source ./launch_scripts/random_top_k.sh cartoon 0.6 $K
rm ./record/random_top_k/cartoon/last.pth
source ./launch_scripts/random_top_k.sh cartoon 0.4 $K
rm ./record/random_top_k/cartoon/last.pth
source ./launch_scripts/random_top_k.sh cartoon 0.2 $K
rm ./record/random_top_k/cartoon/last.pth

source ./launch_scripts/random_top_k.sh photo 1.0 $K
rm ./record/random_top_k/photo/last.pth
source ./launch_scripts/random_top_k.sh photo 0.8 $K
rm ./record/random_top_k/photo/last.pth
source ./launch_scripts/random_top_k.sh photo 0.6 $K
rm ./record/random_top_k/photo/last.pth
source ./launch_scripts/random_top_k.sh photo 0.4 $K
rm ./record/random_top_k/photo/last.pth
source ./launch_scripts/random_top_k.sh photo 0.2 $K
rm ./record/random_top_k/photo/last.pth

source ./launch_scripts/random_top_k.sh sketch 1.0 $K
rm ./record/random_top_k/sketch/last.pth
source ./launch_scripts/random_top_k.sh sketch 0.8 $K
rm ./record/random_top_k/sketch/last.pth
source ./launch_scripts/random_top_k.sh sketch 0.6 $K
rm ./record/random_top_k/sketch/last.pth
source ./launch_scripts/random_top_k.sh sketch 0.4 $K
rm ./record/random_top_k/sketch/last.pth
source ./launch_scripts/random_top_k.sh sketch 0.2 $K
rm ./record/random_top_k/sketch/last.pth



# in case K needs to be changed
# $K=...

# Domain Adaptation experiment, which places ASM after some given layers, and records the performance in each case on target domains
source ./launch_scripts/DA_top_k.sh cartoon conv1 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon maxpool $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon layer1.1.conv2 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon layer2.1.conv2 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon layer3.1.conv2 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon layer4.1.conv2 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon layer1.1.conv2 $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon allConv $K
rm ./record/DA_top_k/cartoon/last.pth
source ./launch_scripts/DA_top_k.sh cartoon avgpool $K
rm ./record/DA_top_k/cartoon/last.pth

source ./launch_scripts/DA_top_k.sh photo conv1 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo maxpool $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo layer1.1.conv2 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo layer2.1.conv2 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo layer3.1.conv2 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo layer4.1.conv2 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo layer1.1.conv2 $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo allConv $K
rm ./record/DA_top_k/photo/last.pth
source ./launch_scripts/DA_top_k.sh photo avgpool $K
rm ./record/DA_top_k/photo/last.pth

source ./launch_scripts/DA_top_k.sh sketch conv1 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch maxpool $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch layer1.1.conv2 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch layer2.1.conv2 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch layer3.1.conv2 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch layer4.1.conv2 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch layer1.1.conv2 $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch allConv $K
rm ./record/DA_top_k/sketch/last.pth
source ./launch_scripts/DA_top_k.sh sketch avgpool $K
rm ./record/DA_top_k/sketch/last.pth

read -n1 -r -p "Press any key to continue..."