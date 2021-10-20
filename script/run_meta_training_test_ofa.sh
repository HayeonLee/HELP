
for run in 1 2 3
do
	python main.py --gpu $1 \
		--search_space ofa \
		--mode 'meta-train' \
		--num_samples 10 \
		--seed $run \
		--num_meta_train_sample 4000 \
		--exp_name 'reproduce_seed_'$run \
        --meta_train_devices '2080ti_1,2080ti_32,2080ti_64,titan_xp_1,titan_xp_32,titan_xp_64,v100_1,v100_32,v100_64' \
        --meta_valid_devices 'titan_rtx_1,titan_rtx_32' \
        --meta_test_devices 'titan_rtx_64,gold_6226' 
done

for run in 1 
do
	python main.py --gpu $1 \
		--search_space ofa \
		--mode 'meta-test' \
		--num_samples 10 \
		--seed $run \
		--num_meta_train_sample 4000 \
		--exp_name 'reproduce_seed_'$run \
		--load_path './results/nasbench201/reproduce_seed_'$run'/checkpoint/help_max_corr.pt' \
        --meta_train_devices '2080ti_1,2080ti_32,2080ti_64,titan_xp_1,titan_xp_32,titan_xp_64,v100_1,v100_32,v100_64' \
        --meta_valid_devices 'titan_rtx_1,titan_rtx_32' \
        --meta_test_devices 'titan_rtx_64,gold_6226' 
done

for run in 1 2 3
do
	python main.py --gpu 0 \
		--search_space ofa \
		--mode 'meta-test' \
		--num_samples 10 \
		--seed $run \
		--num_meta_train_sample 4000 \
		--exp_name 'reproduce_seed_'$run \
		--load_path './data/ofa/checkpoint/help_max_corr.pt' \
        --meta_train_devices '2080ti_1,2080ti_32,2080ti_64,titan_xp_1,titan_xp_32,titan_xp_64,v100_1,v100_32,v100_64' \
        --meta_valid_devices 'titan_rtx_1,titan_rtx_32' \
        --meta_test_devices 'titan_rtx_64' 
done