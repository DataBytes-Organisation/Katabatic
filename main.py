"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
Updated for TensorFlow 2.x: Current date
"""
import os
import datetime
import tensorflow as tf
import sys
import argparse

from model import TableGan
from utils import pp, generate_data, show_all_variables

# Set up argument parser to replace tf.app.flags
parser = argparse.ArgumentParser(description='TableGAN Implementation')
parser.add_argument("--epoch", type=int, default=10, help="Epoch to train [25]")
parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for adam [0.0002]")
parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
parser.add_argument("--train_size", type=int, default=sys.maxsize, help="The size of train images [np.inf]")
parser.add_argument("--y_dim", type=int, default=2, help="Number of unique labels")
parser.add_argument("--batch_size", type=int, default=500, help="The size of batch images [64]")
parser.add_argument("--input_height", type=int, default=16, help="The size of image to use (will be center cropped). [108]")
parser.add_argument("--input_width", type=int, default=None, help="The size of image to use (will be center cropped). If None, same value as input_height [None]")
parser.add_argument("--output_height", type=int, default=16, help="The size of the output images to produce [64]")
parser.add_argument("--output_width", type=int, default=None, help="The size of the output images to produce. If None, same value as output_height [None]")
parser.add_argument("--dataset", type=str, default="celebA", help="The name of dataset [celebA, mnist, lsun]")
parser.add_argument("--checkpoint_par_dir", type=str, default="checkpoint", help="Parent Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--sample_dir", type=str, default="samples", help="Directory name to save the image samples [samples]")
parser.add_argument("--train", action="store_true", default=False, help="True for training, False for testing [False]")
parser.add_argument("--crop", action="store_true", default=False, help="True for training, False for testing [False]")
parser.add_argument("--generate_data", action="store_true", default=False, help="True for visualizing, False for nothing [False]")
parser.add_argument("--alpha", type=float, default=0.5, help="The weight of original GAN part of loss function [0-1.0]")
parser.add_argument("--beta", type=float, default=0.5, help="The weight of information loss part of loss function [0-1.0]")
parser.add_argument("--delta_m", type=float, default=0.5, help="")
parser.add_argument("--delta_v", type=float, default=0.5, help="")
parser.add_argument("--test_id", type=str, default="5555", help="The experiment settings ID.Affecting the values of alpha, beta, delta_m and delta_v.")
parser.add_argument("--label_col", type=int, default=-1, help="The column used in the dataset as the label column (from 0). Used if the Classifer NN is active.")
parser.add_argument("--attrib_num", type=int, default=0, help="The number of columns in the dataset. Used if the Classifer NN is active.")
parser.add_argument("--feature_size", type=int, default=266, help="Size of last FC layer to calculate the Hinge Loss fucntion.")
parser.add_argument("--shadow_gan", action="store_true", default=False, help="True for loading fake data from samples directory[False]")
parser.add_argument("--shgan_input_type", type=int, default=0, help="Input for Discrimiator of shadow_gan. 1=Fake, 2=Test, 3=Train Data")

def main():
    # Parse arguments
    args = parser.parse_args()
    
    a = datetime.datetime.now()

    if args.input_width is None:
        args.input_width = args.input_height
    if args.output_width is None:
        args.output_width = args.output_height

    if not os.path.exists(args.checkpoint_par_dir):
        os.makedirs(args.checkpoint_par_dir)

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    test_cases = [
        {'id': 'OI_11_00', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.0, 'delta_m': 0.0}
        , {'id': 'OI_11_11', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.1, 'delta_m': 0.1}
        , {'id': 'OI_11_22', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.2, 'delta_m': 0.2}

        , {'id': 'OI_101_00', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.0, 'delta_m': 0.0}
        , {'id': 'OI_101_11', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.1, 'delta_m': 0.1}
        , {'id': 'OI_101_22', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.2, 'delta_m': 0.2}

        , {'id': 'OI_1001_00', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.0, 'delta_m': 0.0}
        , {'id': 'OI_1001_11', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.1, 'delta_m': 0.1}
        , {'id': 'OI_1001_22', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.2, 'delta_m': 0.2}
    ]

    found = False
    for case in test_cases:
        if case['id'] == args.test_id:
            found = True
            args.alpha = case['alpha']
            args.beta = case['beta']
            args.delta_m = case['delta_m']
            args.delta_v = case['delta_v']

            print(case)

    if not found:
        print("Using OI_11_00")
        args.test_id = "OI_11_00"
        args.alpha = 1.0
        args.beta = 1.0
        args.delta_m = 0.0
        args.delta_v = 0.0

    args.input_height = 7
    args.input_width = 7
    args.output_height = 7
    args.output_width = 7

    if args.shadow_gan:
        checkpoint_folder = args.checkpoint_par_dir + '/' + args.dataset + "/" + 'atk_' + args.test_id
    else:
        checkpoint_folder = f'{args.checkpoint_par_dir}/{args.dataset}/{args.test_id}'

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    args.checkpoint_dir = checkpoint_folder

    pp.pprint(vars(args))
    print(args.y_dim)

    # TensorFlow 2.x compatibility: Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("Checkpoint: " + args.checkpoint_dir)

    # Create TableGAN model and run training or generation
    # TensorFlow 2.x doesn't use explicit sessions, so we remove the session handling
    tablegan = TableGan(
        input_width=args.input_width,
        input_height=args.input_height,
        output_width=args.output_width,
        output_height=args.output_height,
        batch_size=args.batch_size,
        sample_num=args.batch_size,
        y_dim=args.y_dim,
        dataset_name=args.dataset,
        crop=args.crop,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        alpha=args.alpha,
        beta=args.beta,
        delta_mean=args.delta_m,
        delta_var=args.delta_v,
        label_col=args.label_col,
        attrib_num=args.attrib_num,
        is_shadow_gan=args.shadow_gan,
        test_id=args.test_id
    )

    show_all_variables()

    if args.train:
        tablegan.train(args)
    else:
        if not tablegan.load(args.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")

        # Below is codes for visualization
        if args.shadow_gan:  # using Disriminator sampler for Membership Attack
            OPTION = 5
        else:
            OPTION = 1

        generate_data(tablegan, args, OPTION)

        print('Time Elapsed: ')
        b = datetime.datetime.now()
        print(b - a)


if __name__ == '__main__':
    main()
