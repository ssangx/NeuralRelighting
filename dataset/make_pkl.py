import os
import glob
import pickle
import os.path as osp

"""Save all file name list to pickle for faster loading"""


def get_image_name(albedo_name):
    img_names = glob.glob(albedo_name.replace('albedo', 'image_*'))
    assert len(img_names) == 2
    lights_str = [n.replace('[', '\t').replace(']', '\t').split('\t')[1].replace(',', '') for n in img_names]
    
    lights = []
    for l in lights_str:
        lights.append([float(i) for i in l.split()])

    return img_names, lights


if __name__ == "__main__":

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

    # All images
    images_list = []
    lights_list = []
    for x in albedo_list:
        img_names, lights = get_image_name(x)
        images_list.append(img_names)
        lights_list.append(lights)

    pk_dict = {'albedo_list': albedo_list,
               'normal_list': normal_list,
               'rough_list':  rough_list,
               'depth_list':  depth_list,
               'seg_list':    seg_list,
               'images_list': images_list,
               'lights_list': lights_list}

    print('--> saving pickle file')
    pk_path = (data_root + '_file_list.pickle').replace('SyntheticData', 'DirectionalData')
    with open(pk_path, 'wb') as handle:
        pickle.dump(pk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--> done')