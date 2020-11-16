import os
import glob
import pickle
import numpy as np
import os.path as osp

"""Save all file name list to pickle for faster loading"""


def _get_image_name(albedo_name, pt=True):
    name = albedo_name
    if pt:
        img_names = glob.glob(name.replace('albedo', 'image_pt*'))
    else:
        img_names = glob.glob(name.replace('albedo', 'image_env*'))

    assert len(img_names) == 2, print(albedo_name)
    lights_str = [n.replace('[', '\t').replace(']', '\t').split('\t')[1] for n in img_names]
    
    lights = []
    for l in lights_str:
        lights.append([float(i) for i in l.split(',')])
    assert ([0., 0., 0.] in lights), print(lights)

    if lights[0] == [0., 0., 0.]:
        img_src = img_names[0]
        img_tar = img_names[1]
        light_tar = lights[1]
    elif lights[1] == [0., 0., 0.]:
        img_src = img_names[1]
        img_tar = img_names[0]
        light_tar = lights[0]

    # print('[TEST]', img_src, img_tar, light_tar)
    return img_src, img_tar, np.array(light_tar, dtype=np.float32)


def make_train_set():
    data_root = './data/dataset/Synthetic/train'
    shape_list = glob.glob(osp.join(data_root, 'Shape__*') )
    shape_list = sorted(shape_list)

    albedo_list = []
    for shape in shape_list:
        albedo_names = glob.glob(osp.join(shape, '*albedo.png') )
        albedo_list = albedo_list + albedo_names

    # BRDF parameter
    normal_list = [x.replace('albedo', 'normal') for x in albedo_list]
    rough_list  = [x.replace('albedo', 'rough')  for x in albedo_list]
    seg_list    = [x.replace('albedo', 'seg')    for x in albedo_list]

    # Geometry
    depth_list = [x.replace('albedo', 'depth').replace('png', 'dat') for x in albedo_list]

    # Env
    bg_list = [x.replace('albedo', 'imgEnv') for x in albedo_list]

    # Environment Map
    sh_list = []
    for x in albedo_list:
        suffix = '/'.join(x.split('/')[0:-1])
        fileName = x.split('/')[-1]
        fileName = fileName.split('_')
        sh_list.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'))

    pk_dict = {'albedo_list': albedo_list,
               'normal_list': normal_list,
               'rough_list' : rough_list,
               'depth_list' : depth_list,
               'seg_list'   : seg_list,
               'SH_list'    : sh_list,
               'bg_list'    : bg_list}

    print('--> saving pickle file')
    pk_path = data_root + '_file_list.pickle'
    with open(pk_path, 'wb') as handle:
        pickle.dump(pk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--> done')


def make_eval_set():
    data_root = '/media/ssang/SangShen/Dataset/svbrdf/Data/test'
    shape_list = glob.glob(osp.join(data_root, 'Shape__*') )
    shape_list = sorted(shape_list)

    albedo_list = []
    for shape in shape_list:
        albedo_names = glob.glob(osp.join(shape, '*albedo.png') )
        albedo_list = albedo_list + albedo_names

    # BRDF parameter
    normal_list = [x.replace('albedo', 'normal') for x in albedo_list]
    rough_list  = [x.replace('albedo', 'rough')  for x in albedo_list]
    seg_list    = [x.replace('albedo', 'seg')    for x in albedo_list]

    # Geometry
    depth_list = [x.replace('albedo', 'depth').replace('png', 'dat') for x in albedo_list]

    # Bg
    bgList = [x.replace('albedo', 'imgEnv') for x in albedo_list]

    # Environment Map
    SHList = []
    for x in albedo_list:
        suffix = '/'.join(x.split('/')[0:-1])
        fileName = x.split('/')[-1]
        fileName = fileName.split('_')
        SHList.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'))

    # Rendering
    imagePtSrcList = []
    imagePtTarList = []
    imagePtLightList = []
    for x in albedo_list:
        src_name, tar_name, tar_light = _get_image_name(x, pt=True)
        imagePtSrcList.append(src_name)
        imagePtTarList.append(tar_name)
        imagePtLightList.append(tar_light)

    imageEnvSrcList = []
    imageEnvTarList = []
    imageEnvLightList = []
    for x in albedo_list:
        src_name, tar_name, tar_light = _get_image_name(x, pt=False)
        imageEnvSrcList.append(src_name)
        imageEnvTarList.append(tar_name)
        imageEnvLightList.append(tar_light)

    pk_dict = {'albedo_list': albedo_list,
               'normal_list': normal_list,
               'rough_list' : rough_list,
               'depth_list' : depth_list,
               'seg_list'   : seg_list,
               'SH_list'    : SHList,
               'bg_list'    : bgList,
               'image_pt_src_list': imagePtSrcList,
               'image_pt_tar_list': imagePtTarList,
               'light_pt_tar_list': imagePtLightList,
               'image_env_src_list': imageEnvSrcList,
               'image_env_tar_list': imageEnvTarList,
               'light_env_tar_list': imageEnvLightList}

    print('--> saving pickle file')
    pk_path = data_root + '_file_list.pickle'
    with open(pk_path, 'wb') as handle:
        pickle.dump(pk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--> done')


if __name__ == "__main__":
    # TODO: make pickle for training set or test set
    make_train_set()