target_domain=${1}
layers=${2}
K=${3}

python3 main.py \
--experiment=DA_top_k \
--experiment_name=DA_top_k/${target_domain}/ \
--experiment_args="{'layers_asm' : '${layers}', 'K': ${K}}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=64 \
--num_workers=1 \
--grad_accum_steps=2 \
--epochs=30