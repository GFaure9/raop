import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# from raop.utils import logger


class Graph:
    """
    This class helps to plot graphs from pd.DataFrames obtained from `Option.sensitivity` method.

    Attributes:
        df (pd.DataFrame): either a 2 or 3 columns dataframe. The last column is the output of the graph.
        The first columns are the variables.

    Methods:
        **plot_curve** (`x`, `y`, `save_path`): `x` and `y` are the names of the columns containing the
        data for resp. x and y-axis. If nothing is provided, default is `x`=_first column name_
        and `y`=_second column name_. `save_path` is the path at where the plot will be dumped.
        If nothing is provided, the plot will just be displayed and not saved.

        **plot_surface** (`x`, `y`, `z`, `save_path`): same as plot_curve method but to plot a surface.
        It needs a 3-columns dataframe.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_curve(self, x: str = None, y: str = None, save_path: str = None):
        """
        Create a plot of a curve from `self.df` data, and save it to `save_path` if not None.
        `self.df` must be as follows:

                                | variable_name | output_name |
                                |---------------|-------------|
                                |       x1      |    f(x1)    |
                                |       x2      |    f(x2)    |
                                |       ...     |     ...     |
                                |       xN      |    f(xN)    |

        Args:
            x (str): name of the column in `self.df` for x-axis data. First column if None.
            y (str): name of the column in `self.df` for y-axis data. Second column if None.
            save_path (str): path to save plot. Default value is None.

        Returns:
            None: nothing is returned. The curve is displayed if `save_path` is None.
        """
        if x is None or y is None:
            x, y = self.df.keys()
        x_data, y_data = self.df[x], self.df[y]

        # Plot the figure
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        ax.set_title(f"{y}=f({x})")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.5, ls="--")
        ax.plot(x_data, y_data, label=y, color="blue", linestyle="-", marker="o", markersize=5)
        ax.legend()
        plt.tight_layout()

        # Either save or show the plot
        if save_path is not None:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def plot_surface(self, x: str = None, y: str = None, z: str = None, save_path: str = None):
        """
        Create a plot of a surface from self.df data, and save it to 'save_path' if not None.
        self.df must be as follows:

                    | variable1_name | variable2_name |   output_name   |
                    |--------------- |--------------- |-----------------|
                    |       x1       |       y1       |    f(x1, y1)    |
                    |       x2       |       y2       |    f(x2, y2)    |
                    |       ...      |       ...      |       ...       |
                    |       xN       |       yN       |    f(xN, yN)    |
        Args:
            x (str): name of the column in self.df for x-axis data. First column if None.
            y (str): name of the column in self.df for y-axis data. Second column if None.
            z (str): name of the column in self.df for z-axis data. Third column if None.
            save_path (str): path to save plot. Default value is None.

        Returns:
            None: nothing is returned. The curve is displayed if `save_path` is None.
        """
        if x is None or y is None or z is None:
            x, y, z = self.df.keys()
        x_data, y_data, z_data = self.df[x], self.df[y], self.df[z]

        # Define x, y, z grids
        x_grid = np.linspace(min(x_data), max(x_data), 1000)
        y_grid = np.linspace(min(y_data), max(y_data), 1000)
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = griddata((x_data, y_data), z_data, (xx, yy), method="cubic")

        # Plot the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(xx, yy, zz, cmap="viridis")
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title(f"{z}=f({x}, {y})")

        # Either save or show the plot
        if save_path is not None:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    pass
