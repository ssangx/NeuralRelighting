import glob
import os
import os.path as osp

root = '/media/ssang/SangShen/Dataset/svbrdf/SyntheticData/test'


shapeList = glob.glob(osp.join(root, 'Shape__*'))
shapeList = sorted(shapeList)

for shape in shapeList:
    os.system('rm -r {}/*imgPoint.png'.format(shape))