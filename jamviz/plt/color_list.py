import matplotlib.pyplot as plt

__all__ = ["jcolors"]

# jcolors = ["blue", "orange", "green", "red", "purple", "cyan", "brown"]
# https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
jcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt_colors_names = [
    "aqua",
    "black",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "coral",
    "cornflowerblue",
    "crimson",
    "darkblue",
    "darkgreen",
    "darkorange",
    "darkslateblue",
    "darkslategray",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "firebrick",
    "forestgreen",
    "fuchsia",
    "gray",
    "green",
    "hotpink",
    "lime",
    "maroon",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]


if __name__ == "__main__":
    import matplotlib.patches as mpatch

    num_color = 30
    fig, ax = plt.subplots(1, 1, figsize=(7 * 1, 17 * 1))

    n_rows = 1
    height = 1

    for j, color_name in enumerate(plt_colors_names[:num_color]):
        text_args = dict(fontsize=10, weight="bold")
        ax.add_patch(mpatch.Rectangle((0, j * height), 10, 1, color=color_name))
        ax.text(
            0.5 + 1,
            j * height + 0.7,
            color_name,
            color=color_name,
            ha="center",
            **text_args
        )

    # ax.set_xlim(0, 3 * n_groups)
    ax.set_ylim(num_color, -1)
    ax.axis("off")

    plt.show()
