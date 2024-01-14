# Copyright (c) Alibaba, Inc. and its affiliates.

import os, sys, json
from multiprocessing import Pool

def download_url(item):
    global save_dir
    oss_full_dir = os.path.join("https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse_tar", item+".tar")
    os.system("wget -P {} {}".format(os.path.join(save_dir, item.split("/")[0]), oss_full_dir))

if __name__=="__main__":
    assert len(sys.argv) == 4, "eg: python download_objaverse.py ./data /path/to/json_file 10"
    save_dir = str(sys.argv[1])
    json_file = str(sys.argv[2])
    n_threads = int(sys.argv[3])

    data = json.load(open(json_file))
    p = Pool(n_threads)
    p.map(download_url, data)
