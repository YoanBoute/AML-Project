target_domain=${1}

python main.py \
--experiment=baseline \
--experiment_name=baseline/${target_domain}/ \
--experiment_args="{'layers_asm' : 'conv1', 'ratio_1' : 0.5}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=1 \
--grad_accum_steps=1

read -n1 -r -p "Press any key to continue..."