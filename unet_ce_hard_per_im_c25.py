from subprocess import call

model = 'unet_ce_hard_per_im_c25'

seeds = ['_s0204', '_s1687', '_s2580', '_s3694', '_s4357', '_s5042', '_s6437', '_s7859', '_s8074', '_s9829']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train_unet.py', '--gpu', '1', '--cfg', cfg_name])