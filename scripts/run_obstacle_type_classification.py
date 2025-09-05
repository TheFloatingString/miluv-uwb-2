from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest")
    args = parser.parse_args()
