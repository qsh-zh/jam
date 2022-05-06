import jammy
from jammy import io

parser = jammy.cli.JamArgumentParser()
parser.add_argument("filename", nargs="+")
args = parser.parse_args()


def simple():
    for i, filename in enumerate(args.filename):
        globals()[f"f{i + 1}"] = io.load(filename)

    from IPython import embed

    embed()
