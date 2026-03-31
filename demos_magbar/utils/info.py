# -*- coding: utf-8 -*-
import argparse, sys


def write_down_args(path, parser, args):
    with open(path + "args.txt", "w") as f:
        # Iterate through the attributes of the 'args' object
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")
        f.write("\n")
        # save help info
        original_stdout = sys.stdout
        sys.stdout = f
        parser.print_help()
        sys.stdout = original_stdout

    return None
