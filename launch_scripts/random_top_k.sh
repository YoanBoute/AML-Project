target_domain=${1}
ratio_1=${2}
K=${3}

python3 main.py \
--experiment=random_top_k \
--experiment_name=random_top_k/${target_domain}/ \
--experiment_args="{'ratio_1': ${ratio_1}, 'K': ${K}}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=64 \
--num_workers=1 \
--grad_accum_steps=2 \
--epochs=30