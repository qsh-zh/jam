import jammy
from jammy.utils.notifier import jam_notifier

parser = jammy.cli.JamArgumentParser()
parser.add_argument(
    "text", action="store", type=str, nargs="+", help="The text to send."
)

args = parser.parse_args()

if __name__ == "__main__":
    words = [cur_text.replace('"', '\\"') for cur_text in args.text]
    msg = " ".join(words)
    jam_notifier.notify(msg)
