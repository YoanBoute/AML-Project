target_domain=${1}
layers=${2}

python main.py \
--experiment=DA_no_binarization \
--experiment_name=DA_no_binarization/${target_domain}/ \
--experiment_args="{'layers_asm' : '${layers}'}" \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=1 \
--grad_accum_steps=1 \
--epochs=30