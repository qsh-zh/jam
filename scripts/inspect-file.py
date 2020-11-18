import jammy
import jammy.io as io


parser = jammy.JamArgumentParser()
parser.add_argument('filename', nargs='+')
args = parser.parse_args()


def main():
    for i, filename in enumerate(args.filename):
        globals()[f'f{i + 1}'] = io.load(filename)

    from IPython import embed; embed()


if __name__ == '__main__':
    main()

