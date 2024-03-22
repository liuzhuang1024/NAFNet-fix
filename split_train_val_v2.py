# -*- coding: utf-8 -*-

# Author: Zhuang Liu
# E-mail: 1028741371@qq.com
# Date: 2024-02-08
# Description: 生成数据集以及划分数据集大小

import cv2
import os 
import lmdb
import random
import numpy as np
import math
import tqdm
import pickle



def task(files, idx, val=False):
    crop_size = 768
    # crop_size_h, crop_size_w = 768, 768
    thresh_size = 0
    step = 128

    images = []
    for file in files:

        gt_image_path = os.path.join(root, file)
        lq_image_path = gt_image_path.replace("train_gt", "train_inp")
        # lq_image_path = gt_image_path.replace("train_gt_2k", "train_input_2k")
        gt_image = cv2.imread(gt_image_path, 1)
        lq_image = cv2.imread(lq_image_path, 1)
        
        if val:
            key = f"{file}".split(".")[0] + "png"
            images.append((key, image2bytes(gt_image), image2bytes(lq_image)))
            continue

        h, w, c = gt_image.shape
        
        h_space = np.arange(0, h - crop_size + 1, step)
        if h - (h_space[-1] + crop_size) > thresh_size:
            h_space = np.append(h_space, h - crop_size)
        w_space = np.arange(0, w - crop_size + 1, step)
        if w - (w_space[-1] + crop_size) > thresh_size:
            w_space = np.append(w_space, w - crop_size)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                cropped_img = lq_image[x:x + crop_size, y:y + crop_size, ...]
                _lq_image = np.ascontiguousarray(cropped_img)            

                cropped_img = gt_image[x:x + crop_size, y:y + crop_size, ...]
                _gt_image = np.ascontiguousarray(cropped_img)            
                
                key = f"{index}_{file}".split(".")[0] + ".png"
                images.append((key, image2bytes(_gt_image), image2bytes(_lq_image)))
                index += 1

    pickle.dump(images, open(f"images/{idx}.pkl", "wb"))



def image2bytes(image):
    is_success, im_buf_arr = cv2.imencode(".png", image)
    byte_im = im_buf_arr.tobytes()
    return byte_im


def save(flag="train", path="remove_dataset_768"):
    save_path = f"{flag}_{path}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    metaname = "meta_info.txt"
    lq_path = f"{save_path}/lq.lmdb/"
    gt_path = f"{save_path}/gt.lmdb/"
    gt_txn = lmdb.Environment(gt_path, readonly=False, map_size=int(1<<40)*5, writemap=True).begin(write=True, buffers=True)
    lq_txn = lmdb.Environment(lq_path, readonly=False, map_size=int(1<<40)*5, writemap=True).begin(write=True, buffers=True)
    listnames = []
    for root, _, files in os.walk("images/"):
        for file in tqdm.tqdm(files, desc="Save"):
            file_path = os.path.join(root, file)
            images = pickle.load(open(file_path, "rb"))
            for name, gt_image, lq_image in images:
                print(name, end="\r")
                listnames.append(name)
                gt_txn.put(name.encode(), gt_image)
                lq_txn.put(name.encode(), lq_image)
            shutil.move(file_path, "1.pkl")
    gt_txn.commit()
    lq_txn.commit()
    open(f"{lq_path}/{metaname}", "w").write("\n".join(listnames))
    open(f"{gt_path}/{metaname}", "w").write("\n".join(listnames))

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import threading
    import multiprocessing
    import shutil
    
    pool = ProcessPoolExecutor(128)
    

    listnames = []
    for root, _, files in os.walk("/home/liuzhuang/Dataset/ntire_2024_remove/train_gt"):
        listnames.extend(files)

    random.seed(100)
    random.shuffle(listnames)
    train_list = listnames[:900]
    val_list = listnames[900:]


    step = 10
    # ---------------------------- * ----------------------------
    T = []    
    for i in range(0, len(train_list), step):
        T.append(pool.submit(task, train_list[i:i+step], len(T)))
    for i in tqdm.tqdm(T, "Crop"):
        i.result()
    save("train")
    # ---------------------------- * ----------------------------
    # 验证时候使用整图进行验证
    T = []    
    for i in range(0, len(val_list), step):
        T.append(pool.submit(task, val_list[i:i+step], len(T), val=True))
    for i in tqdm.tqdm(T):
        i.result()
    save("val")
    # ---------------------------- * ----------------------------

    exit("Done!")


