import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


def main(args):
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    hosts = config["server"]["hosts"]
    output_string = " ".join(hosts)
    print(output_string)


if __name__ == "__main__":
    args = parse_args()
    main(args)
