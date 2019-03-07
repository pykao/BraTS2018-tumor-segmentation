from subprocess import call

model = 'unet_ce_hard_per_im_c13'

seeds = ['_s5713', '_s6916', '_s7435', '_s8841', '_s9527']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train_unet.py', '--gpu', '3', '--cfg', cfg_name])
	