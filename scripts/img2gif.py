import glob
from PIL import Image

from jammy.cli.argument import JamArgumentParser

parser = JamArgumentParser()
parser.add_argument("--fmt", required=True, help="glob foramt eg: 'img_*.png'")
parser.add_argument("--out", default="output.gif")
args = parser.parse_args()

# TODO: consider other implementations, https://github.com/wkentaro/video-cli

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
def fp2gif(fp_in, fp_out):
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(
        fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0
    )


if __name__ == "__main__":
    fp2gif(args.fmt, args.out)
