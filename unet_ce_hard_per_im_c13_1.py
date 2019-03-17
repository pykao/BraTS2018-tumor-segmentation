from subprocess import call

model = 'unet_ce_hard_per_im_c13'

seeds = ['_s5042', '_s6437', '_s7859', '_s8074', '_s9829']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train_unet.py', '--gpu', '3', '--cfg', cfg_name])
