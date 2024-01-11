# Copyright (c) Alibaba, Inc. and its affiliates.
# python  /home/joseph/richdreamer/dataset/gobjaverse/download_gobjaverse_280k.py /mnt/Storage/Datasets/gobjaverse_280k gobjaverse_280k.json 16

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
            local_path = os.path.join(save_dir, item, index + "/")
            basename = os.path.basename(oss_full_dir)
            print("local_path", local_path)
            print("remote url", oss_full_dir)
            mkdir_command = "mkdir -p {}".format(local_path)
            os.system(mkdir_command)
            if os.path.exists(os.path.join(local_path, basename)):
                print("existing, skipping")
                continue
            curl_command = "curl -o {} -C - {}".format(os.path.join(local_path, basename + '.tmp'), oss_full_dir)
            print(curl_command)
            os.system(curl_command)
            mv_command = "mv {} {}".format(os.path.join(local_path, basename + '.tmp'), os.path.join(local_path, basename))
            print(mv_command)
            os.system(mv_command)
            # os.system("wget -P {} {}".format(os.path.join(save_dir, item, index + "/"), oss_full_dir))

if __name__=="__main__":
    assert len(sys.argv) == 4, "eg: python ./scripts/data/download_gobjaverse_280k.py ./gobjaverse_280k ./gobjaverse_280k.json 10"
    save_dir = str(sys.argv[1])
    json_file = str(sys.argv[2])
    n_threads = int(sys.argv[3])

    data = json.load(open(json_file))
    p = Pool(n_threads)
    p.map(download_url, data)