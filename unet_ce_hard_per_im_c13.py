from subprocess import call

model = 'unet_ce_hard_per_im_c13'

seeds = ['_s0204', '_s1687', '_s2580', '_s3694', '_s4357']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train_unet.py', '--gpu', '2', '--cfg', cfg_name])
