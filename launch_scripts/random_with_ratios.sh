target_domain=${1}

python main.py \
--experiment=random \
--experiment_name=random/${target_domain}/ \
--experiment_args="{'ratio_1' : 0.}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=1 \
--grad_accum_steps=1 \
--epochs=2

read -n1 -r -p "Press any key to continue..."