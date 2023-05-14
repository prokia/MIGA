import os
import argparse
import warnings

warnings.filterwarnings("ignore", module="matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config',
                        nargs='?',
                        type=str,
                        default='config/miga/_base/iniv.yaml',
                        help='Configuration file to use', )
    parser.add_argument('--local_rank', type=str, default='-1',
                        help='Rank of the current node. (default: %(default)s)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['RANK'] = args.local_rank
    from submiter.network_submiter import NetworkSubmiter
    submiter = NetworkSubmiter(args.config)
    submiter.submit()


if __name__ == '__main__':
    main()
