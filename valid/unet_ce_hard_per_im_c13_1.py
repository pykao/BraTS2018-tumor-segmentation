from subprocess import call

model = 'unet_ce_hard_per_im_c13'

seeds = ['_s0423', '_s1420', '_s2859', '', '_s4867', '_s5713', '_s6916', '_s7435', '_s8841', '_s9527']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'predict_unet.py', '--gpu', '3', '--cfg', cfg_name])
