target_domain=${1}

python main.py \
--experiment=DA \
--experiment_name=DA/${target_domain}/ \
--experiment_args="{'layers_asm' : 'allConv'}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=1 \
--grad_accum_steps=1 \
--epochs=2

read -n1 -r -p "Press any key to continue..."