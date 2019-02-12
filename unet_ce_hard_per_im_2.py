from subprocess import call

model = 'unet_ce_hard_per_im'

seeds = ['_s0423', '_s1420', '_s2859', '', '_s4867']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train_unet.py', '--gpu', '0', '--cfg', cfg_name])
	