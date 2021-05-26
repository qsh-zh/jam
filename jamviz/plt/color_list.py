import matplotlib.pyplot as plt

__all__ = ["jcolors"]

# jcolors = ["blue", "orange", "green", "red", "purple", "cyan", "brown"]
# https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
jcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
