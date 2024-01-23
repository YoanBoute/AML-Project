target_domain=${1}
ratio_1=${2}

python main.py \
--experiment=random_no_binarization \
--experiment_name=random_no_binarization/${target_domain}/ \
--experiment_args="{'ratio_1' : ${ratio_1}}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=1 \
--grad_accum_steps=1 \
--epochs=30