"""SNR Project GUI Module.

Classes to create widgets for SNR program.

Authors: Denis Leahy, Bryson Lawton, Jacqueline Williams
Version: Jan 2019
"""

import tkinter as tk
from tkinter import ttk
import platform

OS = platform.system()

##############################################################################################################################
##############################################################################################################################
class InputParam:
    """Create and grid label widgets for an input parameter.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
    """

    label_options = {"font": "-size 10", "padding": 5}
    input_options = {"font": "-size 10"}
    instances = {}

##############################################################################################################################
    def __init__(self, master, identifier=None, label=None, default=None, padding=True):
        """Create and grid input parameter label.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str, optional): string used as key in instances dictionary
            label (str, optional): input parameter label (includes units)
            default (float/str, optional): default value of parameter
            padding (tuple/int, optional): padding for label, set to default 5 if not specified
        """

        self.identifier = identifier
        self.row = get_row(master)
        if padding != True:
            self.label_options = self.label_options.copy()
            if not padding:
                self.label_options.pop("padding")
            else:
                self.label_options["padding"] = padding
        if label is not None:
            self.label = ttk.Label(master, text=label, **self.label_options)
            self.label.grid(row=self.row, column=0, sticky="w")
        if default is not None:
            self.previous = default
            self.value_var = tk.StringVar(master, str(default))
        if identifier is not None:
            root = "." + master.winfo_parent().split(".")[1]
            if root not in self.instances:
                self.instances[root] = {}
            self.instances[root][identifier] = self

##############################################################################################################################
    def get_value(self):
        """Get current input parameter value.

        Returns:
            float/str: current input parameter value, as a float if possible
        """

        try:
            return float(self.value_var.get())
        except ValueError:
            return self.value_var.get()

##############################################################################################################################
    @classmethod
    def get_values(cls, root):
        """Get all input parameter values.

        Args:
            root (str): ID of window to access widgets from

        Returns:
            dict: current input parameter values for all widgets in root window with valid identifiers
        """

        values = {}
        for identifier, widget in cls.instances[root].items():
            values[identifier] = widget.get_value()
        return values

##############################################################################################################################
##############################################################################################################################
class InputEntry(InputParam):
    """Create entry widget with labels for input parameter.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
        input (ttk.Entry): entry widget for input parameter
        condition (function): function that must return True for input to be valid
    """

    input_options = InputParam.input_options.copy()
    input_options.update({"width": 7})

##############################################################################################################################
    def __init__(self, master, identifier, label, default, callback=None, condition=lambda *args: True, **kwargs):
        """Create and grid input entry widget.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str): string used as key in instances dictionary
            label (str): input parameter label (includes units)
            default (float/str): default value of parameter
            callback (function, optional): function to run when focus leaves widget
            condition (function, optional): condition that must return true for new value to be accepted
            **kwargs: additional options for input widget
        """
        InputParam.__init__(self, master, identifier, label, default, padding=kwargs.pop("padding", True))
        self.input_options = InputEntry.input_options.copy()
        self.input_options.update(**kwargs)
        self.input = ttk.Entry(master, textvariable=self.value_var, **self.input_options, validate="focusout")
        self.input.config(validatecommand=(self.input.register(self.check_input), "%P"))
        self.input.config(invalidcommand=self.revert_value)
        self.input.grid(row=self.row, column=1, sticky="ew")
        if callback is not None:
            self.input.bind("<FocusOut>", lambda e: callback())
        self.input.callback = callback
        self.condition = condition

##############################################################################################################################
    def check_input(self, inp):
        """Check that user input is positive and satisfies any additional conditions specified.

        Args:
            inp (str): user input

        Returns:
            bool: True if user input satisfies conditions, False otherwise
        """
        try:
            current = float(inp)
            if current >= 0 and self.condition(current):
                self.previous = inp
                return True
            else:
                return False
        except ValueError:
            return False

##############################################################################################################################
    def revert_value(self):
        """Revert input to previous acceptable value."""

        self.value_var.set(self.previous)

##############################################################################################################################
##############################################################################################################################
class InputDropdown(InputParam):
    """Create combobox widget (dropdown menu) with labels for input parameter.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
        input (ttk.Combobox): combobox widget for input parameter
    """

    input_options = InputParam.input_options.copy()
    input_options.update({"width": 4, "state": "readonly"})

##############################################################################################################################
    def __init__(self, master, identifier, label, default, callback, values, **kwargs):
        """Create and grid input dropdown widget.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str): string used as key in instances dictionary
            label (str): input parameter label (includes units)
            default (float/str): default value of parameter
            callback (function, optional): function to run when focus leaves widget
            values (tuple): tuple of values for dropdown
            **kwargs: additional options for input widget
        """
        padding = kwargs.pop("padding", True)
        InputParam.__init__(self, master, identifier, label, default, padding=padding)
        self.input_options = InputDropdown.input_options.copy()
        self.input_options.update(kwargs)
        self.input = ttk.Combobox(master, textvariable=self.value_var, values=values, **self.input_options)
        self.input.grid(row=self.row, column=1, sticky="ew")
        self.input.bind("<<ComboboxSelected>>", lambda e: callback())
        self.input.callback = callback

##############################################################################################################################
##############################################################################################################################
class InputRadio(InputParam):
    """Create combobox widget (dropdown menu) with labels for input parameter.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
        input (dict): dictionary of radio button widgets for input parameter
    """

    input_options = {}#InputParam.input_options.copy()
    #input_options.update({"width": 5, "state": "readonly"})

##############################################################################################################################
    def __init__(self, master, identifier, label, default, callback, values, **kwargs):
        """Create and grid input radio button widgets.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str): string used as key in instances dictionary
            label (str): input parameter label (includes units)
            default (float/str): default value of parameter
            callback (function, optional): function to run when focus leaves widget
            values (tuples): tuple of tuples, each associated with a radio button in the form (value, text)
            **kwargs: additional options for input widget
        """

        InputParam.__init__(self, master, identifier, label, default)
        self.input_options = InputRadio.input_options.copy()
        self.input_options.update(**kwargs)
        frame = ttk.Frame(master)
        frame.grid(row=self.row, column=1, sticky="w")
        self.input = {}
        row = ttk.Frame(frame)
        row.pack(fill="x")
        for item in values:
            if "\n" in item:
                row = ttk.Frame(frame)
                row.pack(fill="x")
            self.input[item[0]] = ttk.Radiobutton(row, variable=self.value_var, value=item[0],
                                            text=item[1], **self.input_options)
            self.input[item[0]].pack(side="left")
        #self.input.bind("<FocusOut>", lambda e: callback())
        self.value_var.trace("w", callback)

##############################################################################################################################
##############################################################################################################################
class InputSpinbox(InputEntry):
    """A spinbox widget with a text label.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
        input (ttk.Entry): entry widget for input parameter
    """

    input_options = InputParam.input_options.copy()

##############################################################################################################################
    def __init__(self, master, identifier, label, default, callback, condition=lambda *args: True, **kwargs):
        """Create and grid input spinbox widget.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str): string used as key in instances dictionary
            label (str): input parameter label (includes units)
            default (float/str): default value of parameter
            callback (function, optional): function to run when focus leaves widget
            condition (function, optional): condition that must return true for new value to be accepted
            **kwargs: additional options for input widget
        """

        InputParam.__init__(self, master, identifier, label, default)
        self.input_options = InputSpinbox.input_options.copy()
        self.input_options.update(kwargs)
        self.input = ttk.Entry(master, "ttk::spinbox", textvariable=self.value_var, **self.input_options, validate="focusout")
        self.input.config(validatecommand=(self.input.register(self.check_input), "%P"))
        self.input.config(invalidcommand=self.revert_value)
        self.input.grid(row=self.row, column=1)
        self.input.bind("<FocusOut>", lambda e: callback(self))
        self.input.bind("<<Increment>>", lambda e: callback(self, 1))
        self.input.bind("<<Decrement>>", lambda e: callback(self, -1))
        self.input.callback = callback
        self.condition = condition

##############################################################################################################################
##############################################################################################################################
class InputCheckbox(InputParam):
    """A checkbox widget with a text label.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
        input (ttk.Checkbutton): checkbutton widget for input parameter
    """

##############################################################################################################################
    def __init__(self, master, identifier, label, default, callback, **kwargs):
        """Create and grid input checkbox widget.

        Args:
            master (ttk.Frame): widget container frame
            identifier (str): string used as key in instances dictionary
            label (str): input parameter label (includes units)
            default (float/str): default value of parameter
            callback (function, optional): function to run when focus leaves widget
            **kwargs: additional options for input widget
        """

        InputParam.__init__(self, master, identifier, None, default)
        self.input = ttk.Checkbutton(master, variable=self.value_var, text=label, **kwargs)
        if callback.__name__ == "scale_change":
            self.input.config(command=lambda: callback(self))
        else:
            self.input.config(command=callback)

##############################################################################################################################
##############################################################################################################################
class CheckboxGroup(InputParam):
    """A group of checkbox widgets with a text label.

    Attributes:
        label_options (dict): display options for label
        input_options (dict): display options for input widget
        instances (dict): all InputParam widgets, uses identifiers as keys
        identifier (str): string used as key in instances dictionary
        row (int): row of the container frame in which the widget should be placed
        label (str): input parameter label
        previous (float): widget value that passed validation conditions
        value_var (tk.StringVar): variable to store input parameter value
    """

##############################################################################################################################
    def __init__(self, master, label, callback, values, **kwargs):
        """Create and grid group of checkbox widgets.

        Args:
            master (ttk.Frame): widget container frame
            label (str): input parameter label (includes units)
            callback (function, optional): function to run when focus leaves widget
            values (tuple): tuple of tuples, each associated with a checkbox in the form (identifier, label, default)
            **kwargs: additional options for input widget
        """

        InputParam.__init__(self, master, None, label)
        i = 1
        if type(values[0]) == tuple:
            for item in values:
                InputCheckbox(master, item[0], item[1], item[2], callback, **kwargs).input.grid(row=self.row, column=i)
                i += 1
        else:
            InputCheckbox(master, values[0], values[1], values[2], callback, **kwargs).input.grid(row=self.row, column=i)

##############################################################################################################################
##############################################################################################################################
class DisplayValue:
    """Unchanging displayed value with label and units.

    Attributes:
        options (dict): display options for label
    """

    options = {"font": "-size 10"}

##############################################################################################################################
    def __init__(self, master, label_text, unit, value, digits=4, **kwargs):
        """Create and place widgets needed to display a numerical value with units.

        Args:
            master (ttk.Frame): container frame for widgets
            label_text (str): text for label
            unit (str): text for unit
            value (float): value to display
            digits (int, optional): number of digits to display
            **kwargs: additional display options for label
        """

        row = get_row(master)
        self.options = self.options.copy()
        self.options.update(**kwargs)
        container = ttk.Frame(master)
        container.grid(column=0, row=row, sticky="ew")
        if label_text is not None:
            label = ttk.Label(container, text="%s:" % label_text, **self.options, padding=(5, 2))
            label.pack(side="left")
        formatter = "{0:.{1}g} {2}"
        value = ttk.Label(container, text=formatter.format(value, digits, unit), **self.options)
        value.pack(side="left")

##############################################################################################################################
##############################################################################################################################
class OutputValue:
    """Unchanging displayed value with label and units.

    Attributes:
        options (dict): display options for label
        instances (dict): all OutputValue instances
        value_var (tk.StringVar): variable to store output value
        key (str): string used as key in instances dictionary
        format (str): format string for value
        unit (str): unit associated with value
    """

    options = {"font": "-size 10"}
    instances = {}

##############################################################################################################################
    def __init__(self, master, key, label, unit, digits=4, **kwargs):
        """Create and place widgets needed to display a changeable numerical value with units.

        Args:
            master (ttk.Frame): container frame for widgets
            key (str): identifier for value in instances dictionary
            label (str): text for label
            unit (str): text for unit
            digits (int, optional): number of digits to display
            **kwargs: additional display options for label
        """

        row = get_row(master)
        self.options = self.options.copy()
        self.options.update(**kwargs)
        self.container = ttk.Frame(master)
        self.container.grid(column=0, row=row, columnspan=2, sticky="ew")
        if "padding" in self.options:
            padding = self.options.pop("padding")
        else:
            padding = (5, 1)
        self.label = ttk.Label(self.container, text=label, **self.options, padding=padding)
        self.label.pack(side="left")
        self.value_var = tk.StringVar(master, "")
        if type(padding) == tuple:
            if len(padding) == 2:
                top_padding = bottom_padding = padding[1]
            else:
                top_padding = padding[1]
                bottom_padding = padding[3]
        else:
            top_padding = bottom_padding = padding
        value = ttk.Label(self.container, textvariable=self.value_var, **self.options,
                          padding=(0, top_padding, 0, bottom_padding))
        value.pack(side="left")
        self.key = key
        self.format = '{:.'+str(digits)+'g} {}'
        self.unit = unit
        root = "." + master.winfo_parent().split(".")[1]
        if root not in self.instances:
            self.instances[root] = {}
        self.instances[root][key] = self

##############################################################################################################################
    def update_value(self, source):
        """Update displayed value.

        Args:
            source (dict): dictionary with new value
        """
        try:
            self.value_var.set(self.format.format(source[self.key], self.unit))
        except ValueError:
            self.value_var.set(source[self.key])

##############################################################################################################################
    @classmethod
    def update(cls, source, root, CISM_overwrite, phases=None):
        """Update all OutputValue displayed values.

        Args:
            source (dict): dictionary with new values
            root (str): ID of window containing OutputValue widgets to be updated
            phases (list, optional): phases that occur during evolution of SNR
        """

        for key, widget in cls.instances[root].items():
            if key in source.keys():
                widget.update_value(source)
            if "t-" in key:
                phase = key.split("t-")[1]
                if (phase in phases or ("MRG" in key and phases[-1] in key)):
                    index = phases.index(phase)
                    if (CISM_overwrite == 1 and phases[index-1] == "CISM" and phases[index] == "MCS"):
                        widget.label.config(text="{} to PDS:".format(phases[index-1]))                      
                    elif key != "t-s2":
                        widget.label.config(text="{} to {}:".format(phases[index-1], phases[index]))
                    widget.container.grid()
                elif phase == "MRG" and phases[-1] != "s2":
                    if (CISM_overwrite == 1 and phases[-1] == "MCS"):
                        widget.label.config(text="PDS to merger:")
                    else:
                        widget.label.config(text="{} to merger:".format(phases[-1]))
                    widget.container.grid()
                else:
                    widget.container.grid_remove()

##############################################################################################################################
##############################################################################################################################
class LayoutFrame(ttk.Frame):
    """Frame used to organize widget layout."""

##############################################################################################################################
    def __init__(self, master, padding=0, **grid_options):
        """Create and grid frame widget.

        Args:
            master (ttk.Frame): container for frame
            padding (int/tuple, optional): padding for frame
            **grid_options: options for layout of frame, including row and column if needed
        """

        ttk.Frame.__init__(self, master, padding=padding)
        if "row" not in grid_options:
            grid_options["row"] = get_row(master)
        self.grid(**grid_options, sticky='new')

##############################################################################################################################
##############################################################################################################################
class ScrollWindow:
    """Create window that has automatic scrollbars when needed and centres content on any size of window.

    Attributes:
        root (tk.Tk/tk.Toplevel): tkinter window
        canvas (tk.Canvas): canvas widget used as container to make window scrollable
        vsb (ttk.Scrollbar): vertical scrollbar
        hsb (ttk.Scrollbar): horizontal scrollbar
        container (ttk.Frame): container frame for any input/output widgets in window, centred in window
        window: placed container on canvas widget
    """
##############################################################################################################################
    def __init__(self, window_type=None):
        """Creates empty window with automatic scrollbars.

        Args:
            window_type (str, optional): determines if window should be a Tk or Toplevel window (Tk window must only be
                                         the first window opened, specified by window_type = "root")
        """

        if window_type == "root":
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel()
        self.os = OS
        if OS == "Linux":
            self.root.style = ttk.Style()
            self.root.style.theme_use("clam")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        # Create and grid scrollable canvas
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        # Create and grid vertical and horizontal scrollbars
        self.vsb = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vsb.grid(row=0, column=1, sticky="nse")
        self.vsb.visible = True
        self.hsb = ttk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.hsb.grid(row=1, column=0, sticky="sew")
        self.hsb.visible = True
        self.canvas.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        # Create frame to hold widgets
        wrapper = ttk.Frame(self.canvas)#, relief="groove")
        self.container = ttk.Frame(wrapper)#, relief="ridge", padding=2)
        self.container.place(relx=0.5, rely=0.5, anchor="center")#, relheight=1, relwidth=1)
        self.window = self.canvas.create_window((0, 0), window=wrapper, anchor="nw")
        # Add resize event
        self.root.bind("<Configure>", lambda e: self.window_resize())
        if OS == "Linux":
            self.root.bind("<4>", self.mouse_scroll)
            self.root.bind("<5>", self.mouse_scroll)
        else:
            self.root.bind("<MouseWheel>", self.mouse_scroll)
        self.root.bind("<Down>", lambda e: self.arrow_vscroll(1))
        self.root.bind("<Up>", lambda e: self.arrow_vscroll(-1))
        self.root.bind("<Left>", lambda e: self.arrow_hscroll(-1))
        self.root.bind("<Right>", lambda e: self.arrow_hscroll(1))

##############################################################################################################################
    def window_resize(self):
        """Check scrollbars and reconfigure canvas (run when window is resized)."""

        offset = (self.canvas.winfo_width()-self.container.winfo_reqwidth(),
                  self.canvas.winfo_height()-self.container.winfo_reqheight())
        self.check_scrollbar(self.hsb, offset[0])
        self.check_scrollbar(self.vsb, offset[1])
        self.canvas.itemconfig(
            self.window,
            width=max(self.canvas.winfo_width(), self.container.winfo_reqwidth()),
            height=max(self.canvas.winfo_height(), self.container.winfo_reqheight())
        )

##############################################################################################################################
    def check_scrollbar(self, scrollbar, offset):
        """Check if scrollbars are needed and adjust display accordingly.

        Args:
            scrollbar (ttk.Scrollbar): scrollbar to make visible if needed
            offset (int): difference between actual canvas width and requested canvas width
        """

        if scrollbar.visible and offset >= 0:
            scrollbar.visible = False
            scrollbar.grid_remove()
        elif not scrollbar.visible and offset < 0:
            scrollbar.visible = True
            scrollbar.grid()
            
##############################################################################################################################
    def mouse_scroll(self, event):
        """Scrolls canvas vertically when mouse scrolls.

        Args:
            event: mouse scroll event
        """

        if self.vsb.visible:
            if OS == "Darwin":
                new_delta = -1 * event.delta
            else:
                new_delta = -1 * int(event.delta / 120)
            self.canvas.yview_scroll(new_delta, "units")

##############################################################################################################################
    def arrow_vscroll(self, delta):
        """Scrolls vertically when up or down arrow keys are pressed.

        Args:
            delta (int): number of units to scroll page by
        """

        if self.vsb.visible:
            self.canvas.yview_scroll(delta, "units")

##############################################################################################################################
    def arrow_hscroll(self, delta):
        """Scrolls horizontally when left or right arrow keys are pressed.

        Args:
            delta (int): number of units to scroll page by
        """

        if self.hsb.visible:
            self.canvas.xview_scroll(delta, "units")

##############################################################################################################################
##############################################################################################################################
class SectionTitle(ttk.Label):
    """Bold title for section headings."""

##############################################################################################################################
    def __init__(self, master, title, colspan=1, size=12, **kwargs):
        """Create and grid title.

        Args:
            master (ttk.Frame): container for title widget
            title (str): text for title
            colspan (int, optional): number of columns that title should span
            size (int, optional): font size of title
            **kwargs: additional format options for title
        """

        options = {
            "text": title,
            "font": "-weight bold -size %s" % size
        }
        options.update(kwargs)
        ttk.Label.__init__(self, master, **options)
        row = get_row(master)
        self.grid(row=row, column=0, columnspan=colspan, sticky="new")

##############################################################################################################################
##############################################################################################################################
class SubmitButton(ttk.Button):
    """Button that triggers a function when clicked."""

##############################################################################################################################
    def __init__(self, master, text, action, **kwargs):
        """Create and grid button.

        Args:
            master (ttk.Frame): container frame for button widget
            text (str): text shown in button
            action (function): function to run when button is clicked
            **kwargs: additional display options for button
        """

        ttk.Button.__init__(self, master, text=text, command=action)
        row = get_row(master)
        self.grid(column=0, columnspan=2, row=row, **kwargs)

##############################################################################################################################
def get_row(master):
    """Get first unused row in a given frame widget.

    Args:
        master (ttk.Frame): frame widget to check for an unused row

    Returns:
        int: first unused row of master widget
    """

    try:
        row = int(master.grid_slaves()[0].grid_info()["row"]) + 1
    except IndexError:
        # If master widget does not have any widgets plotted yet
        row = 0
    return row

##############################################################################################################################