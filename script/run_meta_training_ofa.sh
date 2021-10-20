python main.py --gpu $1 \
		--search_space ofa \
		--mode 'meta-train' \
		--num_samples 10 \
		--seed 3 \
		--num_meta_train_sample 4000 \
		--exp_name 'reproduce' \
        --meta_train_devices '2080ti_1,2080ti_32,2080ti_64,titan_xp_1,titan_xp_32,titan_xp_64,v100_1,v100_32,v100_64' \
        --meta_valid_devices 'titan_rtx_1,titan_rtx_32' \
        --meta_test_devices 'titan_rtx_64' 
