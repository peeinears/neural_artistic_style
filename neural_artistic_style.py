#!/usr/bin/env python

import os
import shutil
import subprocess
import copy
import argparse
import numpy as np
import scipy.misc
import deeppy as dp

from matconvnet import vgg_net
from style_network import StyleNetwork


def weight_tuple(s):
    try:
        conv_idx, weight = map(float, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "int,float"')


def float_range(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0, 1]" % x)
    return x


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


def imread(path):
    return scipy.misc.imread(path).astype(dp.float_)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def to_bc01(img):
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def to_rgb(img):
    return np.transpose(img[0], (1, 2, 0))


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', required=True, type=str,
                        help='Subject image or video.')
    parser.add_argument('--style', required=True, type=str,
                        help='Style image.')
    parser.add_argument('--output', default='out.png', type=str,
                        help='Output file.')
    parser.add_argument('--init', default=None, type=str,
                        help='Initial image. Subject is chosen as default.')
    parser.add_argument('--init-noise', default=0.0, type=float_range,
                        help='Weight between [0, 1] to adjust the noise level '
                             'in the initial image.')
    parser.add_argument('--size', default=None, type=int,
                        help='Resize width before styling.')
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Random state.')
    parser.add_argument('--animation', dest="animation_enabled",
                        action="store_true", help='Enables animation.')
    parser.add_argument('--animation-directory', default='animation', type=str,
                        help='Output animation directory.')
    parser.add_argument('--animation-rate', default=1, type=int,
                        help='Number of iterations between each frame.')
    parser.add_argument('--iterations', default=500, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--learn-rate', default=2.0, type=float,
                        help='Learning rate.')
    parser.add_argument('--smoothness', type=float, default=5e-8,
                        help='Weight of smoothing scheme.')
    parser.add_argument('--subject-weights', nargs='*', type=weight_tuple,
                        default=[(9, 1)],
                        help='List of subject weights (conv_idx,weight).')
    parser.add_argument('--style-weights', nargs='*', type=weight_tuple,
                        default=[(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                        help='List of style weights (conv_idx,weight).')
    parser.add_argument('--subject-ratio', type=float, default=2e-2,
                        help='Weight of subject relative to style.')
    parser.add_argument('--pool-method', default='avg', type=str,
                        choices=['max', 'avg'], help='Subsampling scheme.')
    parser.add_argument('--network', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='Network in MatConvNet format).')
    parser.add_argument('--framerate', default=12, type=int,
                        help='Frame rate for video (fps).')

    args = parser.parse_args()

    subject_file_extension = os.path.splitext(args.subject)[1].lower()
    if subject_file_extension == '.mp4' or subject_file_extension == '.mov':
        style_video(args)
    else:
        style_image(args)

def style_video(args):
    input_frames_dir = 'input_frames'
    output_frames_dir = 'output_frames'
    audio_file = 'raw-audio.wav'

    # make sure we're starting from empty frames dirs
    if os.path.exists(input_frames_dir):
        shutil.rmtree(input_frames_dir)
    if os.path.exists(output_frames_dir):
        shutil.rmtree(output_frames_dir)

    os.mkdir(input_frames_dir)
    os.mkdir(output_frames_dir)

    split_frames_cmd = [
            'ffmpeg',
            '-i', args.subject,
            '-r', str(args.framerate),
            '-f', 'image2']

    if args.frames_scale is not None:
        split_frames_cmd.extend(['-vf', 'scale=' + args.frames_scale ])

    split_frames_cmd.append(os.path.join(input_frames_dir, 'frame-%5d.jpg'))

    extract_audio_cmd = ['ffmpeg', '-y', '-i', args.subject, audio_file]

    subprocess.call(split_frames_cmd)
    subprocess.call(extract_audio_cmd)

    frame_args = copy.copy(args)
    frame_args.animation = False

    frames = os.listdir(input_frames_dir)
    for i, frame in enumerate(frames):
        print('Styling frame %i of %i' % (i + 1, len(frames)))
        frame_args.subject = os.path.join(input_frames_dir, frame)
        frame_args.output = os.path.join(output_frames_dir, frame)
        style_image(frame_args)

    # strip out extension to replace with .mp4
    output = os.path.splitext(args.output)[0] + '.mp4'

    make_video_cmd = [
            'ffmpeg',
            '-framerate', str(args.framerate),
            '-y', # overwrite output files without asking
            '-i', os.path.join(output_frames_dir, 'frame-%05d.jpg'),
            '-i', audio_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output]

    subprocess.call(make_video_cmd)


def style_image(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    layers, pixel_mean = vgg_net(args.network, pool_method=args.pool_method)

    # Inputs
    style_img = imread(args.style) - pixel_mean
    subject_img = imread(args.subject)

    if args.size is not None:
        # resize maintaining aspect ratio
        width, height = subject_img.shape[:2]
        new_width = args.size
        new_height = height * new_width / width
        subject_img = scipy.misc.imresize(subject_img, (new_width,  new_height))

    subject_img = subject_img - pixel_mean

    if args.init is None:
        init_img = subject_img
    else:
        init_img = imread(args.init) - pixel_mean
    noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
    init_img = init_img * (1 - args.init_noise) + noise * args.init_noise

    # Setup network
    subject_weights = weight_array(args.subject_weights) * args.subject_ratio
    style_weights = weight_array(args.style_weights)
    net = StyleNetwork(layers, to_bc01(init_img), to_bc01(subject_img),
                       to_bc01(style_img), subject_weights, style_weights,
                       args.smoothness)

    # Repaint image
    def net_img():
        return to_rgb(net.image) + pixel_mean

    if args.animation_enabled and not os.path.exists(args.animation_directory):
        os.mkdir(args.animation_directory)

    params = net.params
    learn_rule = dp.Adam(learn_rate=args.learn_rate)
    learn_rule_states = [learn_rule.init_state(p) for p in params]
    for i in range(args.iterations):
        if args.animation_enabled and i % args.animation_rate == 0:
            imsave(os.path.join(args.animation_directory, '%.4d.png' % i), net_img())
        cost = np.mean(net.update())
        for param, state in zip(params, learn_rule_states):
            learn_rule.step(param, state)
        print('Iteration: %i, cost: %.4f' % (i, cost))
    imsave(args.output, net_img())


if __name__ == "__main__":
    run()
