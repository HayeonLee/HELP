
python main.py --gpu $1 \
		--search_space nasbench201 \
		--mode nas \
		--num_samples 10 \
		--seed 3 \
		--num_meta_train_sample 900 \
		--load_path './data/nasbench201/checkpoint/help_max_corr.pt' \
		--sampled_arch_path 'data/nasbench201/arch_generated_by_metad2a.txt' \
		--nas_target_device 'gold_6226' \
		--latency_constraint '11.0' \
		--exp_name 'nas' \
		--meta_train_devices '1080ti_1,1080ti_32,1080ti_256,silver_4114,silver_4210r,samsung_a50,pixel3,essential_ph_1,samsung_s7' \
		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
		--meta_test_devices 'titan_rtx_256,gold_6226,fpga,pixel2,raspi4,eyeriss'
