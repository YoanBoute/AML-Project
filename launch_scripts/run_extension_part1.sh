# set K for all the experiments


# Random experiment, which adds an ASM after each layer of the network, and tests the performance on target domains with random maps containing between 20% and 100% of 1
source ./launch_scripts/random_no_binarization.sh cartoon 1.0 
rm ./record/random_no_binarization/cartoon/last.pth
source ./launch_scripts/random_no_binarization.sh cartoon 0.8 
rm ./record/random_no_binarization/cartoon/last.pth
source ./launch_scripts/random_no_binarization.sh cartoon 0.6 
rm ./record/random_no_binarization/cartoon/last.pth
source ./launch_scripts/random_no_binarization.sh cartoon 0.4 
rm ./record/random_no_binarization/cartoon/last.pth
source ./launch_scripts/random_no_binarization.sh cartoon 0.2 
rm ./record/random_no_binarization/cartoon/last.pth

source ./launch_scripts/random_no_binarization.sh photo 1.0 
rm ./record/random_no_binarization/photo/last.pth
source ./launch_scripts/random_no_binarization.sh photo 0.8 
rm ./record/random_no_binarization/photo/last.pth
source ./launch_scripts/random_no_binarization.sh photo 0.6 
rm ./record/random_no_binarization/photo/last.pth
source ./launch_scripts/random_no_binarization.sh photo 0.4 
rm ./record/random_no_binarization/photo/last.pth
source ./launch_scripts/random_no_binarization.sh photo 0.2 
rm ./record/random_no_binarization/photo/last.pth

source ./launch_scripts/random_no_binarization.sh sketch 1.0 
rm ./record/random_no_binarization/sketch/last.pth
source ./launch_scripts/random_no_binarization.sh sketch 0.8 
rm ./record/random_no_binarization/sketch/last.pth
source ./launch_scripts/random_no_binarization.sh sketch 0.6 
rm ./record/random_no_binarization/sketch/last.pth
source ./launch_scripts/random_no_binarization.sh sketch 0.4 
rm ./record/random_no_binarization/sketch/last.pth
source ./launch_scripts/random_no_binarization.sh sketch 0.2 
rm ./record/random_no_binarization/sketch/last.pth



# in case K needs to be changed
# =...

# Domain Adaptation experiment, which places ASM after some given layers, and records the performance in each case on target domains
source ./launch_scripts/DA_no_binarization.sh cartoon conv1 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon maxpool 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon layer1.1.conv2 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon layer2.1.conv2 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon layer3.1.conv2 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon layer4.1.conv2 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon layer1.1.conv2 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon allConv 
rm ./record/DA_no_binarization/cartoon/last.pth
source ./launch_scripts/DA_no_binarization.sh cartoon avgpool 
rm ./record/DA_no_binarization/cartoon/last.pth

source ./launch_scripts/DA_no_binarization.sh photo conv1 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo maxpool 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo layer1.1.conv2 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo layer2.1.conv2 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo layer3.1.conv2 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo layer4.1.conv2 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo layer1.1.conv2 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo allConv 
rm ./record/DA_no_binarization/photo/last.pth
source ./launch_scripts/DA_no_binarization.sh photo avgpool 
rm ./record/DA_no_binarization/photo/last.pth

source ./launch_scripts/DA_no_binarization.sh sketch conv1 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch maxpool 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch layer1.1.conv2 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch layer2.1.conv2 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch layer3.1.conv2 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch layer4.1.conv2 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch layer1.1.conv2 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']" 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch allConv 
rm ./record/DA_no_binarization/sketch/last.pth
source ./launch_scripts/DA_no_binarization.sh sketch avgpool 
rm ./record/DA_no_binarization/sketch/last.pth

read -n1 -r -p "Press any key to continue..."