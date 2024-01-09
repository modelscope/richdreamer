# Copyright (c) Alibaba, Inc. and its affiliates.

import os, sys, json
from multiprocessing import Pool

def download_url(item):
    end = 40 # hard-coded
    copy_items = ['.json','.png','_albedo.png','_hdr.exr','_mr.png','_nd.exr','_ng.exr'] # hard-coded
    global save_dir
    oss_base_dir = os.path.join("https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse", item, "campos_512_v4")
    for index in range(end):
        index = "{:05d}".format(index)
        for copy_item in copy_items:
            postfix = index + "/" + index + copy_item
            oss_full_dir = os.path.join(oss_base_dir, postfix)
            print(oss_full_dir)
            os.system("wget -P {} {}".format(os.path.join(save_dir, item, index + "/"), oss_full_dir))

if __name__=="__main__":
    assert len(sys.argv) == 4, "eg: python ./scripts/data/download_gobjaverse_280k.py ./gobjaverse_280k ./gobjaverse_280k.json 10"
    save_dir = str(sys.argv[1])
    json_file = str(sys.argv[2])
    n_threads = int(sys.argv[3])

    data = json.load(open(json_file))
    p = Pool(n_threads)
    p.map(download_url, data)