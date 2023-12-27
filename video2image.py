""" video2image
"""
from __future__ import division, print_function

import argparse
import glob
import multiprocessing
import os
import os.path as osp
import subprocess
from functools import partial

n_thread = 50


def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser(description='Rescale videos')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('out_path', type=str,
                        help='root directory for the out videos')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3],
                        help='the number of level for folders')
    parser.add_argument('--lib', type=str, default='ffmpeg',
                        choices=['opencv', 'ffmpeg'],
                        help='the decode lib')
    parser.add_argument('-se', '--short_edge', type=int, default=None)
    parser.add_argument('-fps', type=int, default=None)
    parser.add_argument('--prefix', type=str, default='img_%05d.jpg')
    args = parser.parse_args()
    return args


def vid2jpg(tup, decode_type, se=None, fps=None, prefix='img_%05d.jpg'):
    """[summary]

    Args:
        tup ([type]): [description]
        decode_type ([type]): [description]
        se ([type], optional): [description]. Defaults to None.
        fps ([type], optional): [description]. Defaults to None.
        prefix (str, optional): [description]. Defaults to 'img_%05d.jpg'.
    """
    src, dest = tup
    folder, name = osp.split(dest)
    video_name = name.split('.')[0]
    video_folder = osp.join(folder, video_name)
    try:
        if osp.exists(video_folder):
            if not osp.exists(
                    osp.join(video_folder, prefix.format(1))):
                subprocess.call(
                    'rm -r \"{}\"'.format(video_folder), shell=True)
                print('remove {}'.format(video_folder))
                os.system('mkdir -p {}'.format(video_folder))
            else:
                print('*** convert has been done: {}'.
                      format(video_folder))
                return
        else:
            os.system('mkdir -p {}'.format(video_folder))
    except Exception:
        print(video_folder)
        return

    import cv2
    cap = cv2.VideoCapture(src)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width, height = int(w), int(h)

    if decode_type == 'ffmpeg':
        if se is None:
            if fps is None:
                cmd = 'ffmpeg -i \"{}\"  -threads 1 -q:v 1 \
                    \"{}/{}\"'.format(src, video_folder, prefix)
            else:
                cmd = 'ffmpeg -i \"{}\"  -threads 1 -q:v 1 -r {} \
                    \"{}/{}\"'.format(src, fps, video_folder, prefix)
        else:
            if width > height:
                if fps is None:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale=-1:{} -q:v 1 \
                        \"{}/{}\"'.format(src, se, video_folder, prefix)
                else:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale=-1:{} -q:v 1 -r {} \
                        \"{}/{}\"'.format(src, se, fps, video_folder, prefix)
            else:
                if fps is None:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale={}:-1 -q:v 1 \
                        \"{}/{}\"'.format(src, se, video_folder, prefix)
                else:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale={}:-1 -q:v 1 -r {} \
                        \"{}/{}\"'.format(src, se, fps, video_folder, prefix)
        # print(cmd)
        subprocess.call(cmd, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif decode_type == 'opencv':
        frame_list = []
        import cv2
        cap = cv2.VideoCapture(src)
        ret, frame = cap.read()
        if ret:
            while ret:
                if se is not None:
                    if width > height:
                        frame = cv2.resize(frame, (int(height / se * width), se))
                    else:
                        frame = cv2.resize(frame, (se, int(width / se * height)))
                frame_list.append(frame)
                ret, frame = cap.read()
            for i in range(1, len(frame_list)):
                out_img = '{}/{}'.format(video_folder, prefix % i)
                cv2.imwrite(out_img, frame_list[i])
        else:
            if se is None:
                cmd = 'ffmpeg -i \"{}\"  -threads 1 -q:v 1 \
                    \"{}/{}\"'.format(src, video_folder, prefix)
            else:
                if width > height:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale=-1:{} -q:v 1 \
                        \"{}/{}\"'.format(src, se, video_folder, prefix)
                else:
                    cmd = 'ffmpeg -i \"{}\"  -threads 1 -vf scale={}:-1 -q:v 1 \
                        \"{}/{}\"'.format(src, se, video_folder, prefix)
            # print(cmd)
            subprocess.call(cmd, shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def main():
    """main
    """
    args = parse_args()

    if args.level == 1:
        src_list = glob.glob(osp.join(args.video_path, '*'))
        dest_list = [osp.join(args.out_path, osp.split(vid)[-1])
                     for vid in src_list]
        # ['root/xxx.mp4']
    elif args.level == 2:
        src_list = glob.glob(osp.join(args.video_path, '*', '*'))
        dest_list = [osp.join(
            args.out_path, vid.split('/')[-2], osp.split(vid)[-1])
            for vid in src_list]
        # ['root/class/xxx.mp4']
    elif args.level == 3:
        src_list = glob.glob(osp.join(args.video_path, '*', '*', '*'))
        dest_list = [osp.join(
            args.out_path, vid.split('/')[-3],
            vid.split('/')[-2], osp.split(vid)[-1])
            for vid in src_list]
        # ['root/class/sub/xxx.mp4']

    vid_list = list(zip(src_list, dest_list))

    pool = multiprocessing.Pool(n_thread)
    worker = partial(vid2jpg, decode_type=args.lib,
                     se=args.short_edge, fps=args.fps, prefix=args.prefix)
    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(worker, vid_list),
                  total=len(vid_list)):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
