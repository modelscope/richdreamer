import os, json, argparse
from tqdm import tqdm


def download_url(ulr, save_dir):
    command = 'wget -P {} {}'.format(os.path.join(save_dir,ulr.split('/')[-2]), ulr)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download gobjaverse alignment.')
    parser.add_argument('--json_path', type=str, required=True ,default='gobjaverse_alignment.json', help='json file')
    parser.add_argument('--save_dir', type=str, required=True, default='save_dir/', help='save dir')

    args = parser.parse_args()

    with open(args.json_path,'r') as f:
        urls = json.load(f)

    for url in tqdm(urls):
        download_url(url, args.save_dir)



