from subprocess import call

model = 'deepmedic_ce_c13_45_60_75_b50_mb50_all'

seeds = ['_s5112', '_s6428', '_s7926', '_s8846', '_s9527']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train.py', '--gpu', '0', '--cfg', cfg_name])