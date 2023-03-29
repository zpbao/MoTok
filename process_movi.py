import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

# first need to download the movi dataset to 
dataset_dir = '/data/MOVI-tfds/'
out_dir = '/data/MOVI/movi-e/train/'
# loop among [train, validation, test]

ds, ds_info = tfds.load("movi_e", data_dir=dataset_dir, with_info=True)
# or use:
# ds, info = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", with_info=True) 
train_iter = iter(tfds.as_numpy(ds["train"]))

example = next(train_iter)
count = 0
while example:
    minv, maxv = example["metadata"]["forward_flow_range"]
    forward_flow = example["forward_flow"] / 65535 * (maxv - minv) + minv

    minv, maxv = example["metadata"]["backward_flow_range"]
    backward_flow = example["backward_flow"] / 65535 * (maxv - minv) + minv

    minv, maxv = example["metadata"]["depth_range"]
    depth = example["depth"] / 65535 * (maxv - minv) + minv

    rgb = example["video"]
    segment = example["segmentations"]

    video_name = 'video_{}'.format(str(count).zfill(4))

    np.save(out_dir + video_name+'/rgb.npy', rgb)
    np.save(out_dir + video_name+'/forward_flow.npy', forward_flow)
    np.save(out_dir + video_name+'/backward_flow.npy', backward_flow)
    np.save(out_dir + video_name+'/depth.npy', depth)
    np.save(out_dir + video_name+'/segment.npy', segment)
    if count%200 == 0:
        print(count)
    count+=1
    example = next(train_iter)


