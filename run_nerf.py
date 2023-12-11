import argparse
import os
import sys
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="an astronaut riding a horse")
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--iters", "-i", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--img_res", "-r", type=int, default=128)
    parser.add_argument("--save_mem", "-s", type=int, default=0)
    args, extras = parser.parse_known_args()

    return args, extras


def main(args, extras):
    start_time = time.time()

    if args.iters is not None:
        cmd = f'bash scripts/nerf/run_single.sh {args.gpus} "{args.text}" {args.output_dir} {args.img_res} {args.save_mem} trainer.max_steps={args.iters}'
    else:
        cmd = f'bash scripts/nerf/run_single.sh {args.gpus} "{args.text}" {args.output_dir} {args.img_res} {args.save_mem}'
    print(f"cmd:{cmd}")
    os.system(cmd)
    print(f"time cost:{time.time() - start_time}")


if __name__ == "__main__":
    args, extras = get_args()
    main(args, extras)
