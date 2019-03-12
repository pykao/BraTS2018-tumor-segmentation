import nibabel as nib
import os
import numpy as np

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    #print('thuyen', data.dtype)
    #data = data.astype('float32')
    proxy.uncache()
    return data

#suffix = 'HarvardOxford-sub'
suffix = 'VOI-1mm'

# training
#mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/HarvardOxford-sub/training'
#mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/VOI-1mm/train'
#root = '/media/hdd1/pkao/brats2018/training'
#flist = 'all.txt'

# validation
#mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/HarvardOxford-sub/validation/'
mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/VOI-1mm/valid'
root = '/media/hdd1/pkao/brats2018/validation'
flist = 'test.txt'

# testing
#mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/HarvardOxford-sub/testing/'
#root = '/usr/data/pkao/brats2018/testing'
#flist = 'test.txt'

flist = open(os.path.join(root, flist)).read().splitlines()
names = [x.split('/')[-1] for x in flist]

for k, name in enumerate(names):
    oname = os.path.join(root, flist[k], name + '_' + suffix + '.npy')
    iname = os.path.join(mask_dir, name + '_' + suffix + '.nii.gz')

    img = np.array(nib_load(iname), dtype='uint8', order='C')

    np.save(oname, img)
