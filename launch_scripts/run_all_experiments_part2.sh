# Domain Adaptation experiment, which places ASM after some given layers, and records the performance in each case on target domains
source ./launch_scripts/DA_with_layers.sh cartoon conv1
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon maxpool
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon layer1.1.conv2
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon layer2.1.conv2
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon layer3.1.conv2
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon layer4.1.conv2
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon layer1.1.conv2
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']"
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon allConv
rm ./record/DA/cartoon/last.pth
source ./launch_scripts/DA_with_layers.sh cartoon avgpool
rm ./record/DA/cartoon/last.pth

source ./launch_scripts/DA_with_layers.sh photo conv1
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo maxpool
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo layer1.1.conv2
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo layer2.1.conv2
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo layer3.1.conv2
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo layer4.1.conv2
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo layer1.1.conv2
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']"
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo allConv
rm ./record/DA/photo/last.pth
source ./launch_scripts/DA_with_layers.sh photo avgpool
rm ./record/DA/photo/last.pth

source ./launch_scripts/DA_with_layers.sh sketch conv1
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch maxpool
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch layer1.1.conv2
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch layer2.1.conv2
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch layer3.1.conv2
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch layer4.1.conv2
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch layer1.1.conv2
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch "['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2']"
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch allConv
rm ./record/DA/sketch/last.pth
source ./launch_scripts/DA_with_layers.sh sketch avgpool
rm ./record/DA/sketch/last.pth

read -n1 -r -p "Press any key to continue..."