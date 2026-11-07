import argparse


def parse_args(desc: str):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    ap.add_argument("--smoke", action="store_true", help="run on the tiny synthetic corpus (no download)")
    return ap.parse_args()
