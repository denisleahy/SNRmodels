"""SNR Project Plotting Module.

Classes used to create MatPlotLib plots with custom scientific notation display and simple update functions.

Authors: Denis Leahy, Bryson Lawton, Jacqueline Williams
Version: Jan 2019
"""

import math
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

##############################################################################################################################
##############################################################################################################################
class OutputPlot(Figure):
    """Create a simple plot with automatic y-scaling.

    Attributes:
        xlabel (str): x-axis title
        ylabel (str): y-axis title
        graph: subplot instance for plot
        ticker (mpl.ticker.ScalarFormatter): formatter for x- and y-axis ticks
        canvas (FigureCanvasTkAgg): container for plot
    """
##############################################################################################################################
    def __init__(self, master, size, xlabel, ylabel, **kwargs):
        """Create empty plot and initialize options.

        Args:
            master (snr_gui.LayoutFrame): container frame widget
            size (tuple): width and height of plot in inches
            xlabel (str): x-axis title
            ylabel (str): y-axis title
            **kwargs: additional options for plot
        """

        Figure.__init__(self, figsize=size, facecolor="white")
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.graph = self.add_subplot(111, xlabel=self.xlabel, ylabel=self.ylabel, **kwargs)
        self.ticker = mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
        self.graph.xaxis.set_major_formatter(self.ticker)
        self.graph.yaxis.set_major_formatter(self.ticker)
        self.graph.xaxis.get_major_formatter().set_powerlimits((-4,4))
        self.graph.yaxis.get_major_formatter().set_powerlimits((-4,4))
        self.graph.xaxis.labelpad = 2
        self.graph.yaxis.labelpad = 2
        self.canvas = FigureCanvasTkAgg(self, master=master)
        self.canvas.draw() #show() in other versions of Python
        self.canvas.get_tk_widget().grid(column=0, row=0)
        # Remove scientific notation from edges of axes (will be added to axis titles when plot limits are known)
        self.graph.xaxis.offsetText.set_visible(False)
        self.graph.yaxis.offsetText.set_visible(False)

##############################################################################################################################
    def add_data(self, x_data, y_data, **kwargs):
        """Add line to plot.

        Args:
            x_data (np.ndarray): x-values for line
            y_data (np.ndarray): y-values associated with x-values in x_data
            **kwargs: additional options for plot
        """

        self.graph.plot(x_data, y_data, **kwargs)

##############################################################################################################################
    def display_plot(self, top=1, limits=None):
        """Display current plot.

        Args:
            top (float, optional): normalized height of plot area
            limits (tuple, optional): x-limits of plot, autoscaled if not provided
        """

        self.graph.relim()
        if limits is not None:
            self.graph.set_xlim(*limits)
            x_scale = False
        else:
            x_scale = True
        self.graph.autoscale_view(None, x_scale, True)
        self.tight_layout(rect=(0, 0, 1, top), pad=0.5)  # First tight_layout needed to get proper scientific notation
        x_sci = self.graph.xaxis.get_offset_text().get_text().replace("\\times", "") + " "
        y_sci = self.graph.yaxis.get_offset_text().get_text().replace("\\times", "") + " "
        # Show scientific notation in axis titles
        x_label = self.xlabel if x_sci == " " else self.xlabel.replace("/", "/{}".format(x_sci))
        y_label = self.ylabel if y_sci == " " else self.ylabel.replace("/", "/{}".format(y_sci))
        self.graph.set_xlabel(x_label)
        self.graph.set_ylabel(y_label)
        self.tight_layout(rect=(0, 0, 1, top), pad=0.5)  # Second tight-layout adjusts for the new axis titles
        self.canvas.draw() #show() in other version of Python

##############################################################################################################################
    def clear_plot(self):
        """Remove all lines from plot."""
        self.graph.lines.clear()

##############################################################################################################################
    def update_plot(self):
        pass

##############################################################################################################################
##############################################################################################################################
class TimePlot(OutputPlot):
    """Specific type of OutputPlot used for radius/velocity vs. time graphs. Includes more advanced automatic y-scaling
    to account for a plotted x-range smaller than the range of data given.

    Attributes:
        ylabel_dict (dict): dictionary of y-axis titles for different plot types
        xlabel (str): x-axis title
        ylabel (str): y-axis title
        graph: subplot instance for plot
        ticker (mpl.ticker.ScalarFormatter): formatter for x- and y-axis ticks
        canvas (FigureCanvasTkAgg): container for plot
    """

##############################################################################################################################
    def __init__(self, master, size):
        """Create empty plot and initialize options.

        Args:
            master (snr_gui.LayoutFrame): container frame widget
            size (tuple): width and height of plot in inches
        """

        self.ylabel_dict = {"v": r"Velocity/km s$^{-1}$", "r": "Radius/pc", "eMeas": "Emission Measure/cm^-3", "temper": "Temperature/K"}
        # Set default to radius graph
        self.ylabel = self.ylabel_dict["r"]
        OutputPlot.__init__(self, master, size, "Time/yr", self.ylabel)

##############################################################################################################################
    def display_plot(self, time_limit):
        """Display current plot.

        Args:
            time_limit (tuple, optional): x-limits of plot
        """

        # All visible lines
        vis_lines = []
        # Visible lines excluding phase-transition lines
        data_lines = []
        # Labels of the visible lines
        labels = []
        if self.graph.get_xscale() == "log" and time_limit[0] < 1:
            # Prevent lower time limit of less than 1 for log scale
            time_limit = (1, time_limit[1])
        self.graph.set_xlim(*time_limit)
        for line in self.graph.lines:
            if line.get_xdata()[-1] > min(time_limit) and line.get_xdata()[0] < max(time_limit):
                vis_lines.append(line)
                labels.append(line.get_label())
                if line.get_c() != "black":
                    data_lines.append(line)
        try:
            x_data = np.concatenate([line.get_xdata() for line in data_lines])
            y_data = np.concatenate([line.get_ydata() for line in data_lines])
            lower_edge = [np.interp(time_limit[0], line.get_xdata(), line.get_ydata()) for line in data_lines]
            upper_edge = [np.interp(time_limit[1], line.get_xdata(), line.get_ydata()) for line in data_lines]
            visible_y = np.concatenate([y_data[np.where(np.logical_and(
                x_data >= time_limit[0], x_data <= time_limit[1]))], lower_edge, upper_edge])
            if self.graph.get_yscale() == "log" and "Radius" in self.graph.get_ylabel():
                if time_limit[0] <= time_limit[1]:
                    self.graph.set_ylim(min(lower_edge), np.amax(visible_y))
                else:
                    self.graph.set_ylim(min(upper_edge), np.amax(visible_y))
            else:
                self.graph.set_ylim(np.amin(visible_y), np.amax(visible_y))
        except ValueError:
            # Catch cases where the range doesn't include any lines
            pass
        if self.graph.get_yscale() == "log":
            # Prevent cases where the y-axis has no visible tick labels
            ylim = self.graph.get_ylim()
            if math.floor(math.log10(ylim[0])) == math.floor(math.log10(ylim[1])):
                ylim = (10.0 ** (math.floor(math.log10(ylim[0]))), 10.0 ** (math.floor(math.log10(ylim[0])) + 1))
                self.graph.set_ylim(ylim)
        self.graph.legend(vis_lines, labels, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.11), frameon=False,
                          ncol=5, columnspacing=0.4, handletextpad=0)
        OutputPlot.display_plot(self, 0.95)

##############################################################################################################################
    def update_title(self, plot_type):
        """Update y-axis label on plot.

        Args:
            plot_type (str): r if radius vs. time plot, v if velocity vs. time plot
        """

        self.ylabel = self.ylabel_dict[plot_type]

##############################################################################################################################