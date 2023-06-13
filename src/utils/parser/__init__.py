import argparse


def init_parser():
    parser = argparse.ArgumentParser('Fix-Prop-Repair-Learn',
                                     description='implementation of Fix-Prop-Repair-Learn')
    subparser = parser.add_subparsers(description='solver command',
                                      required=True)

    # learning parser
    learn_parser = subparser.add_parser('learn',
                                        help='train network')
    learn_parser.add_argument('instance_dir',
                              help='directory contain training instances')
    solve_parser = subparser.add_parser('solve',
                                        help='find solution to MIP instance')

    return parser
