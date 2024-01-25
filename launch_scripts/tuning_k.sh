for K in 1, 10, 100, 1000, 10000, 10000
do
    source ./launch_scripts/random_top_k.sh photo 1.0 $K
    #rm ./record/random/photo/last.pth #impossible to remove
done