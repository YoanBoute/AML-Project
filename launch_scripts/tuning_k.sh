for K in 10 20 30 40 50 #max 50
do
    echo "@@@@@" $K
    source ./launch_scripts/random_top_k.sh photo 1.0 $K
    rm ./record/random_top_k/photo/last.pth #impossible to remove
done