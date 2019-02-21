from subprocess import call

model = 'deepmedic_ce_c13_45_60_75_b50_mb50_all'

seeds = ['_s0104', '_s1716', '_s2114', '_s3204', '_s4251']

for seed in seeds:
	cfg_name = model+seed
	call(['python', 'train.py', '--gpu', '3', '--cfg', cfg_name])
