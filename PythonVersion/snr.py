"""SNR Project Main Program.

Creates GUI and initializes case-specific functions.

Authors: Denis Leahy, Bryson Lawton, Jacqueline Williams
Version: Jan 2019
"""

import snr_gui as gui
import snr_calc as calc
import snr_plot as plt
import snr_fileCalc as fileCalc
import math

# To create executable:
# pyinstaller --noconfirm --log-level=ERROR filename.spec

ELEMENT_NAMES = {"H": "Hydrogen", "He": "Helium", "O": "Oxygen", "C": "Carbon", "Ne": "Neon", "N": "Nitrogen",
                 "Mg": "Magnesium", "Si": "Silicon", "Fe": "Iron", "S": "Sulphur", "Ca": "Calcium", "Ni": "Nickel", 
                 "Na": "Sodium", "Al": "Aluminium", "Ar": "Argon"}
ELEMENT_ORDER = ["He", "O", "C",  "Ne", "N", "Mg", "Si", "Fe", "S", "Ca", "Ni", "Na", "Al", "Ar"]
MODEL_DICT = {"fel": "Fractional energy loss", "standard": "Standard", "hld": "Hot low-density media", "cism": "Cloudy ISM", "sedtay": "Sedov-Taylor"}
ABUNDANCE = {"Solar": {"H": 12, "He": 10.93, "O": 8.69, "C": 8.43, "Ne": 7.93, "N": 7.83, "Mg": 7.60, "Si": 7.51,
                       "Fe": 7.50, "S": 7.12, "Ca": 6.34, "Ni": 6.22, "Na": 6.24, "Al": 6.45, "Ar": 6.40},

             "LMC": {"H": 12, "He": 10.94, "O": 8.35, "C": 8.04, "Ne": 7.61, "N": 7.14, "Mg": 7.47, "Si": 7.81,
                       "Fe": 7.23, "S": 6.70, "Ca": 6.04, "Ni": 5.92, "Na": 5.94, "Al": 6.15, "Ar": 6.10},
                       
             "Type Ia": {"H": 12, "He": 10.93, "O": 12.69, "C": 0, "Ne": 12.654, "N": 0, "Mg": 11.962, "Si": 12.872,
                       "Fe": 13.133, "S": 12.518, "Ca": 11.973, "Ni": 11.853, "Na": 6.24, "Al": 6.45, "Ar": 11.815},
                         
            "CC": {"H": 12, "He": 11.216, "O": 9.548, "C": 9.163, "Ne": 8.773, "N": 8.527, "Mg": 8.324, "Si": 8.746,
                       "Fe": 8.554, "S": 8.331, "Ca": 7.483, "Ni": 7.363, "Na": 7.383, "Al": 7.593, "Ar": 7.543}
             } 
switchPlotToDefault = False     
##############################################################################################################################
def get_model_name(key, snr_em):
    """Get name of model to be shown on emissivity window.

    Args:
        key (str): short string representing model used, see MODEL_DICT keys for possible values and associated models
        snr_em (calc.SNREmissivity): supernova remnant emissivity class instance

    Returns:
        str: name of current emissivity model
    """

    name = MODEL_DICT[key]
    if name == "Standard":
        if snr_em.data["model"] == "chev":
            name = "Standard (s\u200a=\u200a{0:.0f}, n\u200a=\u200a{1})".format(SNR.data["s"], SNR.data["n"])
        else:
            name = "Standard (Sedov)"
    elif (name == "Cloudy ISM" or name == "Sedov-Taylor"):
        if SNR.data["t"] < SNR.calc["t_st"]:
            name = "Standard (s\u200a=\u200a{0:.0f}, n\u200a=\u200a{1})".format(SNR.data["s"], SNR.data["n"])
        else:
            name = "{0} (C/\u03C4\u200a=\u200a{1:.0f})".format(name, SNR.data["c_tau"])
    return name

##############################################################################################################################
def s_change(update=True):
    """Changes available input parameters when s is changed.

    Args:
        update (bool): true if SuperNovaRemnant instance needs to be updated (generally only False when run during
                       initialization to avoid running update_output multiple times unnecessarily)
    """

    if widgets["s"].value_var.get() == '0':
        # Remove unnecessary parameters m_w and v_w and change available n values
        widgets["m_w"].input.grid_remove()
        widgets["m_w"].label.grid_remove()
        widgets["v_w"].input.grid_remove()
        widgets["v_w"].label.grid_remove()
        widgets["n_0"].input.grid()
        widgets["n_0"].label.grid()
        widgets["sigma_v"].input.grid()
        widgets["sigma_v"].label.grid()
        if ((widgets["n"].get_value() >= 6) and (widgets["model"].get_value() == "standard")):
            widgets["plot_type"].input["eMeas"].config(state="normal")
            widgets["plot_type"].input["temper"].config(state="normal")
        widgets["model"].input["fel"].config(state="normal")
        widgets["model"].input["cism"].config(state="normal")
        widgets["model"].input["sedtay"].config(state="normal")
        if 0.1 * SNR.calc["t_c"] <= SNR.calc["t_pds"]:
            widgets["model"].input["hld"].config(state="normal")
        widgets["n"].input.config(values=(0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14)) 
        if widgets["n"].get_value() == 1:
            # Reset n value if old value not available
            widgets["n"].value_var.set(0)
    else:
        global switchPlotToDefault
        # Restore previous values for input parameters m_w and v_w and reduce available n values
        widgets["m_w"].input.grid()
        widgets["m_w"].label.grid()
        widgets["v_w"].input.grid()
        widgets["v_w"].label.grid()
        widgets["n_0"].input.grid_remove()
        widgets["n_0"].label.grid_remove()
        widgets["sigma_v"].input.grid_remove()
        widgets["sigma_v"].label.grid_remove()
        if (widgets["plot_type"].get_value() == "eMeas" or widgets["plot_type"].get_value() == "temper"):
            switchPlotToDefault = True
            widgets["plot_type"].value_var.set("r")
        widgets["plot_type"].input["eMeas"].config(state="disabled")
        widgets["plot_type"].input["temper"].config(state="disabled")
        widgets["model"].input["fel"].config(state="disabled")
        widgets["model"].input["hld"].config(state="disabled")
        widgets["model"].input["cism"].config(state="disabled")
        widgets["model"].input["sedtay"].config(state="disabled")
        widgets["n"].input.config(values=(6, 7, 8, 9, 10, 11, 12, 13, 14))
        if widgets["n"].get_value() not in (6, 7, 8, 9, 10, 11, 12, 13, 14):
            # Reset n value if old value not available
            widgets["n"].value_var.set(6)
    if update:
        SNR.update_output()

##############################################################################################################################
def n_change(update=True):
    """Changes available input parameters when n is changed.

    Args:
        update (bool): true if SuperNovaRemnant instance needs to be updated (generally only False when run during
                       initialization to avoid running update_output multiple times unnecessarily)
    """
    if ((widgets["n"].get_value() >= 6) and (widgets["s"].get_value() == 0) and (widgets["model"].get_value() == "standard")):
        widgets["plot_type"].input["eMeas"].config(state="normal")
        widgets["plot_type"].input["temper"].config(state="normal")
    else:
        global switchPlotToDefault
        if (widgets["plot_type"].get_value() == "eMeas" or widgets["plot_type"].get_value() == "temper"):
            switchPlotToDefault = True
            widgets["plot_type"].value_var.set("r")   
        widgets["plot_type"].input["eMeas"].config(state="disabled")
        widgets["plot_type"].input["temper"].config(state="disabled") 
 
    if update:
        SNR.update_output()

##############################################################################################################################
def title_change():
    """Change plot y-label and replot when plot type is changed without updating and recomputing values."""
    global switchPlotToDefault
    plot_type = widgets["plot_type"].get_value()
    SNR.graph.update_title(plot_type)
    SNR.data["plot_type"] = plot_type
    if (not switchPlotToDefault):
        SNR.update_plot(SNR.get_phases())
    else:
        switchPlotToDefault = False

##############################################################################################################################
def enter_pressed(event):
    """Validate input and update output when user presses the enter key.

    Args:
        event: event object passed by tkInter <Return> event
    """

    if hasattr(event.widget, "validate"):
        event.widget.validate()
        # Check if widget is one of the axis limit spinboxes
        if event.widget is not widgets["xmin"].input and event.widget is not widgets["xmax"].input:
            event.widget.callback()
        else:
            # If widget is a spinbox, only update limits rather than running the full update function
            if event.widget is widgets["xmin"].input:
                widget = widgets["xmin"]
            else:
                widget = widgets["xmax"]
            increment_xlimits(widget, 0)

##############################################################################################################################
def limit_change():
    """Update plot when preset limits change without running the full update function."""

    SNR.data["range"] = SNR.widgets["range"].get_value()
    SNR.graph.display_plot(SNR.get_limits())

##############################################################################################################################
def increment_xlimits(widget, direction=0):
    """Update plot when custom limits change without running the full update function.

    Args:
        widget: widget object that had its value changed
        direction: 0 typed value, +1 for value increased with clicked arrow/arrow keys, -1 for value decreased with
                   clicked arrow/arrow keys
    """

    # Change dropdown value to read custom if not already set
    if SNR.data["range"] != "Custom":
        SNR.data["range"] = "Custom"
        SNR.widgets["range"].value_var.set("Custom")
    old_val = widget.get_value()
    try:
        # Update increment value to appropriate power of 10 or set the increment to 10 if the value is less than 10
        increment = max(10 ** (math.floor(math.log10(old_val))), 10)
        if old_val == increment and increment != 10 and direction == -1:
            # Decrease the increment if the value is being decreased by the user
            increment /= 10
    except ValueError:
        # If old_val = 0, set the increment to 10
        increment = 10
    widget.input.config(increment=increment)
    if direction != 0:
        # Manually change x-limit stored in SNR if arrows are used since widget value updates after this function runs
        maximum = widget.input.cget("to")
        new_val = old_val + direction * widget.input.cget("increment")
        if new_val % increment != 0 and not (round(old_val) == round(maximum) and direction == 1):
            # Change value to a multiple of the increment if the arrows or arrow keys were used to change the value
            # If statement prevents this from happening at the maximum value
            if direction == 1:
                new_val = math.floor(new_val/increment)*increment
            else:
                new_val = math.ceil(new_val/increment)*increment
            widget.value_var.set(new_val - direction * increment)
        if new_val > maximum:
            # Prevents spinbox from going over maximum allowed value
            new_val = maximum
        if not (old_val == 0 and direction == -1) and not (new_val == widgets["xmin"].get_value() or
                                                           new_val == widgets["xmax"].get_value()):
            # Update SNR stored value unless the value will be lower than the minimum value of 0
            SNR.data[widget.identifier] = new_val
            widget.previous = round(new_val)
    else:
        SNR.data[widget.identifier] = old_val
        widget.previous = round(old_val)
    # Redraw plot
    SNR.graph.display_plot(SNR.get_limits())

##############################################################################################################################
def model_change(update=True):
    """Update available input parameters when model type is changed and recalculate output.

    Args:
        update (bool): true if SuperNovaRemnant instance needs to be updated (generally only False when run during
                       initialization to avoid running update_output multiple times unnecessarily)
    """
    global switchPlotToDefault
    if widgets["model"].get_value() != "fel":
        widgets["t_fel"].input.grid_remove()
        widgets["t_fel"].label.grid_remove()
        widgets["gamma_1"].input.grid_remove()
        widgets["gamma_1"].label.grid_remove()
    else:
        widgets["t_fel"].input.validate()
        widgets["t_fel"].input.grid()
        widgets["t_fel"].label.grid()
        widgets["gamma_1"].input.grid()
        widgets["gamma_1"].label.grid()
       
        
    if widgets["model"].get_value() == "standard":
        widgets["s"].input.config(state="readonly")
        if ((widgets["n"].get_value() >= 6) and (widgets["s"].get_value() == 0)):
            widgets["plot_type"].input["eMeas"].config(state="normal")
            widgets["plot_type"].input["temper"].config(state="normal")
        else:
            if (widgets["plot_type"].get_value() == "eMeas" or widgets["plot_type"].get_value() == "temper"):
                switchPlotToDefault = True
                widgets["plot_type"].value_var.set("r")

            widgets["plot_type"].input["eMeas"].config(state="disabled")
            widgets["plot_type"].input["temper"].config(state="disabled") 
    else:
        if (widgets["plot_type"].get_value() == "eMeas" or widgets["plot_type"].get_value() == "temper"):
            switchPlotToDefault = True
            widgets["plot_type"].value_var.set("r")

        widgets["plot_type"].input["eMeas"].config(state="disabled")
        widgets["plot_type"].input["temper"].config(state="disabled") 
        widgets["s"].input.config(state="disabled")


    if widgets["model"].get_value() == "hld":
        widgets["t_hld"].input.config(state="normal")
        widgets["t_hld"].revert_value()
        widgets["t_hld"].input.grid()
        widgets["t_hld"].label.grid()
    else:
        widgets["t_hld"].input.config(state="disabled")
        widgets["t_hld"].value_var.set("N/A")
        widgets["t_hld"].input.grid_remove()
        widgets["t_hld"].label.grid_remove()


    if widgets["model"].get_value() == "cism":
        widgets["c_tau"].input.grid()
        widgets["c_tau"].label.grid()
    else:
        widgets["c_tau"].input.grid_remove()
        widgets["c_tau"].label.grid_remove()


    if widgets["model"].get_value() == "sedtay":
        widgets["n"].value_var.set("0")
        widgets["n"].input.config(state="disabled")
        widgets["c_tau"].value_var.set(0)
        widgets["m_ej"].input.grid_remove()
        widgets["m_ej"].label.grid_remove()
    else:
        widgets["n"].input.config(state="readonly")
        widgets["c_tau"].value_var.set(1)
        widgets["m_ej"].input.grid()
        widgets["m_ej"].label.grid()

    if update:
        SNR.update_output()

##############################################################################################################################
def scale_change(widget):
    """Trigger change between linear and log scales on the plot axes.

    Args:
        widget: checkbox that triggered the function
    """

    if widget.identifier == "y_scale":
        if widget.get_value() == 1:
            SNR.graph.graph.set_yscale("log")
        else:
            SNR.graph.graph.set_yscale("linear")
            SNR.graph.graph.yaxis.set_major_formatter(SNR.graph.ticker)
    else:
        if widget.get_value() == 1:
            SNR.graph.graph.set_xscale("log")
        else:
            SNR.graph.graph.set_xscale("linear")
            SNR.graph.graph.xaxis.set_major_formatter(SNR.graph.ticker)
    SNR.graph.display_plot(SNR.get_limits())

##############################################################################################################################
def update_ratio():
    dropdown = widgets["T_ratio"]
    if dropdown.get_value() == "Default":
        dropdown.input.config(state="readonly")
        dropdown.input.config(values="Custom")
    else:
        dropdown.input.config(state="normal")
        dropdown.input.config(values="Default")
    SNR.update_output()

##############################################################################################################################
def abundance_window(abundance_dict, ab_type):
    """Create window to view and adjust element abundances.

    Args:
        abundance_dict (dict): dictionary with default/current element abundances
        ab_type (str): type of abundance window, "ejecta" or "ISM"
    """

    if ab_window_open[ab_type]:
        # Give focus to existing window rather than opening a second window
        ab_window_open[ab_type].root.focus()
    else:
        window = gui.ScrollWindow()
        ab_window_open[ab_type] = window
        window.root.focus()
        window.root.geometry("%dx%d+%d+%d" %(200, 400, APP.root.winfo_x(), APP.root.winfo_y()))
        frame = window.container
        gui.SectionTitle(frame, "Element", size=10)
        if window.os != "Windows":
            title = "log(X/H)+12"
        else:
            title = "log(X/H)\u200a+\u200a12"
        gui.SectionTitle(gui.LayoutFrame(frame, column=1, row=0), title, size=10)
        for element in ELEMENT_ORDER:
            entry = gui.InputEntry(frame, element, ELEMENT_NAMES[element], "{0:.2f}".format(abundance_dict[element]),
                                   condition=lambda value: 0 < value < 100, padding=(0, 0, 5, 0))
            entry.input.bind(
                "<Key>", lambda *args: gui.InputParam.instances[str(window.root)]["ab_type"].value_var.set("Custom"))
        button_frame = gui.LayoutFrame(frame, columnspan=2, padding=(0, 10))
        
        if ab_type == "ISM":
            types = ("Solar", "LMC")
            default_type = ism_ab_type
        else:
            types = ("Type Ia", "CC")
            default_type = ej_ab_type
            
        gui.InputDropdown(gui.LayoutFrame(button_frame, row=0, column=0, padding=(2, 1, 2, 0)), "ab_type", None,
                          default_type, lambda: reset_ab(str(window.root)), types, width=7)
        gui.SubmitButton(gui.LayoutFrame(button_frame, row=0, column=1), "Submit",
                         lambda: ab_window_close(window.root, abundance_dict, ab_type))
        window.root.bind("<1>", lambda event: event.widget.focus_set())
        window.root.bind("<Return>", lambda event: ab_window_close(window.root, abundance_dict, ab_type, event))
        window.root.protocol("WM_DELETE_WINDOW", lambda: ab_window_close(window.root, abundance_dict, ab_type))
        window.root.update()
        window.canvas.config(scrollregion=(0, 0, window.container.winfo_reqwidth(), window.container.winfo_reqheight()))

##############################################################################################################################
def reset_ab(root_id):
    """Reset abundance window to defaults.

    Args:
        root_id (str): id of abundance window, used to access input widgets
    """

    elements = gui.InputParam.instances[root_id].copy()
    ab_default = elements.pop("ab_type").get_value()
    for element, widget in elements.items():
        widget.value_var.set("{0:.2f}".format(ABUNDANCE[ab_default][element]))

##############################################################################################################################
def ab_window_close(root, ab_dict, ab_type, event=None):
    """Close abundance window and update abundance related variables.

    Args:
        root: tkInter TopLevel window instance for abundance window to be closed
        ab_dict (dict): dictionary of abundance values
        ab_type (str): type of abundance window, either "ejecta" or "ISM"
        event: event returned by tkInter if window was closed using the enter key
    """

    global ism_ab_type, ej_ab_type
    if event and hasattr(event.widget, "validate"):
        event.widget.validate()
    ab_window_open[ab_type] = False
    ab_dict.update(gui.InputParam.get_values(str(root)))
    
    # Store current ejecta type from dropdown
    if ab_type == "ISM":
        ism_ab_type = ab_dict.pop("ab_type")
    else:
        ej_ab_type = ab_dict.pop("ab_type")
    root.destroy()
    SNR.update_output()
    SNR_INV.update_output()
    
##############################################################################################################################
def inverse_window():
    """Create window for inverse mode."""
    global inverseWindowActive
    global forward_mode
    global SNR_INV
    global inverseWindow
    
    if (inverseWindowActive == True):
        inverseWindow.root.focus()
        return
    else:
        inverseWindowActive = True
        inverseWindow = gui.ScrollWindow()
        inverseWindow.root.focus()
        inverseWindow.root.geometry("%dx%d+%d+%d" %(900, 400, (ws-900)/2, (hs-400)/2))
        inverseWindow.root.update()
    
        root_id_inv = str(inverseWindow.root)
        SNR_INV = calc.SuperNovaRemnantInverse(root_id_inv)
        gui.InputParam.instances[root_id_inv] = {}
        widgets = gui.InputParam.instances[root_id_inv]
    
        inverseFrame = inverseWindow.container
        inverseWindow.input_frame = gui.LayoutFrame(inverseFrame, 2, row=0, column=0)
        gui.SectionTitle(inverseWindow.input_frame, "Inverse Mode Input", 2)
        gui.InputDropdown(inverseWindow.input_frame, "ctau_inv", "C/\u03C4", 0, SNR_INV.update_output, (0, 1, 2, 4))
        gui.InputDropdown(inverseWindow.input_frame, "s_inv", "CSM power-law index, s:", 0, lambda: s_change_inv(widgets, SNR_INV), (0, 2))
        gui.InputDropdown(inverseWindow.input_frame, "n_inv", "Ejecta power-law index, n:", 7, SNR_INV.update_output,
            (6, 7, 8, 9, 10, 11, 12, 13, 14))
    
        #Inverse Input Parameters
        gui.SectionTitle(inverseWindow.input_frame, "Observed Shock Input Parameters:", 2)
        gui.InputEntry(inverseWindow.input_frame, "m_eject_inv", "Ejected mass (M\u2609):", 1.2, SNR_INV.update_output, gt_zero)
        gui.InputEntry(inverseWindow.input_frame, "R_f_inv", "Forward Shock Radius (pc):", 3.32, SNR_INV.update_output, gt_zero)    
        gui.InputEntry(inverseWindow.input_frame, "Te_f_inv", "Forward Shock X-ray\nElectron Temperature (keV):", 1.06, SNR_INV.update_output, gt_zero)
        gui.InputEntry(inverseWindow.input_frame, "EM58_f_inv", "Forward Shock X-ray\nEmission Measure (x 10\u2075\u2078 cm\u207B\u00B3):", 0.9459, SNR_INV.update_output, gt_zero) 
       
        gui.InputEntry(inverseWindow.input_frame, "Te_r_inv", "Reverse Shock X-ray\nElectron Temperature (keV):", 1.06, SNR_INV.update_output, gt_zero)
        gui.InputEntry(inverseWindow.input_frame, "EM58_r_inv", "Reverse Shock X-ray\nEmission Measure (x 10\u2075\u2078 cm\u207B\u00B3):", 0.9459, SNR_INV.update_output, gt_zero) 
    
        
        gui.InputParam(inverseWindow.input_frame, label="Model type:")
    
        MODEL_FRAME_INV = gui.LayoutFrame(inverseWindow.input_frame, 0, row=100, column=0, columnspan=2)
        gui.InputRadio(MODEL_FRAME_INV, "model_inv", None, "standard_forward", lambda *args: model_change_inv(widgets, SNR_INV),
                   (("standard_forward", "Standard Forward Shock", "\n"), ("standard_reverse", "Standard Reverse Shock", "\n"), 
                    ("cloudy_forward", "Cloudy Forward Shock (m\u2091\u2C7C = 0)", "\n"),
                    ("sedov_forward", "Sedov (T\u2091 = T\u1D62)", "\n")), padding=(10, 0, 0, 0))
        
        inverseWindow.output_frame = gui.LayoutFrame(inverseFrame, 2, row = 0, column=1, columnspan=2)
        
    
        inverseWindow.output_values = gui.LayoutFrame(inverseWindow.output_frame, (20, 0, 0, 0), row=0, column=0)
        inverseWindow.output_times = gui.LayoutFrame(inverseWindow.output_frame, (20, 0, 0, 0), row=0, column=1)
    
        gui.SectionTitle(inverseWindow.output_values, "Output\nCalculated values:", 2)
        gui.SectionTitle(inverseWindow.output_times, "\nPhase transition times:", 2)
        gui.OutputValue(inverseWindow.output_values, "t_inv", "", "yrs")
        gui.OutputValue(inverseWindow.output_values, "E51_inv", "", "x 10\u2075\u00B9 erg")
        gui.OutputValue(inverseWindow.output_values, "n_0_inv", "", "cm\u207B\u00B3", padding=(0, 0, 10, 10))
        gui.OutputValue(inverseWindow.output_values, "R_f_out", "  Forward shock radius:", "pc")
        gui.OutputValue(inverseWindow.output_values, "t_f_out", "  Forward shock electron temperature:", "keV")
        gui.OutputValue(inverseWindow.output_values, "EM_f_out", "   Forward shock emission measure:", "x 10\u2075\u2078 cm\u207B\u00B3", padding=(0, 0, 0, 10))
        gui.OutputValue(inverseWindow.output_values, "R_r_out", "  Reverse shock radius:", "pc")
        gui.OutputValue(inverseWindow.output_values, "t_r_out", "  Reverse shock electron temperature:", "keV")
        gui.OutputValue(inverseWindow.output_values, "EM_r_out", "  Reverse shock emission measure:", "x 10\u2075\u2078 cm\u207B\u00B3")
        gui.OutputValue(inverseWindow.output_times, "t-s2", "", "", padding=0)
        gui.OutputValue(inverseWindow.output_times, "Core", "", "", padding=0)
        gui.OutputValue(inverseWindow.output_times, "Rev", "", "", padding=0)
        
        gui.SubmitButton(inverseWindow.output_values, "Run Inverse Calc on Datafile", lambda: fileCalc.openFileBrowser(inverseWindow, SNR_INV), sticky="w", padx=15, pady=(10, 10))
        
        model_change_inv(widgets, SNR_INV, False),
        
        inverseWindow.root.bind("<1>", lambda event: event.widget.focus_set())
        inverseWindow.root.protocol("WM_DELETE_WINDOW", lambda: close_inverse_window(inverseWindow.root, SNR_INV))
        SNR_INV.update_output()
        inverseWindow.root.update()
        inverseWindow.canvas.config(scrollregion=(0, 0, inverseWindow.container.winfo_reqwidth(), inverseWindow.container.winfo_reqheight()))

##############################################################################################################################
def close_inverse_window(root, SNR_INV):
    global inverseWindowActive
    inverseWindowActive = False
    SNR_INV.update_output()
    root.destroy()
    
##############################################################################################################################
def model_change_inv(widgets, SNR_INV, update=True):
    """Changes SNRPY to inverse mode

    Args:
        update (bool): true if SuperNovaRemnant instance needs to be updated (generally only False when run during
                       initialization to avoid running update_output multiple times unnecessarily)
    """

    if widgets["model_inv"].get_value() == "standard_forward":
        widgets["s_inv"].input.config(state="normal")
        widgets["n_inv"].input.config(state="normal")
        widgets["m_eject_inv"].input.config(state="normal")
        widgets["m_eject_inv"].value_var.set("1.2")
        widgets["ctau_inv"].input.grid_remove()
        widgets["ctau_inv"].label.grid_remove()
        widgets["s_inv"].input.grid()
        widgets["s_inv"].label.grid()
        widgets["n_inv"].input.grid()
        widgets["n_inv"].label.grid()
        
            
        widgets["Te_r_inv"].input.grid_remove()
        widgets["Te_r_inv"].label.grid_remove()
        widgets["EM58_r_inv"].input.grid_remove()
        widgets["EM58_r_inv"].label.grid_remove()
        
        widgets["Te_f_inv"].input.grid()
        widgets["Te_f_inv"].label.grid()
        widgets["EM58_f_inv"].input.grid()
        widgets["EM58_f_inv"].label.grid()
           
        
    elif widgets["model_inv"].get_value() == "standard_reverse":
        widgets["s_inv"].input.config(state="normal")
        widgets["n_inv"].input.config(state="normal")
        widgets["m_eject_inv"].input.config(state="normal")
        widgets["m_eject_inv"].value_var.set("1.2")
        widgets["ctau_inv"].input.grid_remove()
        widgets["ctau_inv"].label.grid_remove()
        widgets["s_inv"].input.grid()
        widgets["s_inv"].label.grid()
        widgets["n_inv"].input.grid()
        widgets["n_inv"].label.grid()
        
        widgets["Te_f_inv"].input.grid_remove()
        widgets["Te_f_inv"].label.grid_remove()
        widgets["EM58_f_inv"].input.grid_remove()
        widgets["EM58_f_inv"].label.grid_remove()
        
        widgets["Te_r_inv"].input.grid()
        widgets["Te_r_inv"].label.grid()
        widgets["EM58_r_inv"].input.grid()
        widgets["EM58_r_inv"].label.grid()
    
    elif widgets["model_inv"].get_value() == "cloudy_forward":
        
        widgets["m_eject_inv"].value_var.set("0")
        widgets["m_eject_inv"].input.config(state="disabled")
        widgets["ctau_inv"].input.grid()
        widgets["ctau_inv"].label.grid()
        widgets["s_inv"].value_var.set("0")
        widgets["s_inv"].input.grid_remove()
        widgets["s_inv"].label.grid_remove()
        widgets["n_inv"].input.grid_remove()
        widgets["n_inv"].label.grid_remove()
        
        widgets["Te_r_inv"].input.grid_remove()
        widgets["Te_r_inv"].label.grid_remove()
        widgets["EM58_r_inv"].input.grid_remove()
        widgets["EM58_r_inv"].label.grid_remove()
        
        widgets["Te_f_inv"].input.grid()
        widgets["Te_f_inv"].label.grid()
        widgets["EM58_f_inv"].input.grid()
        widgets["EM58_f_inv"].label.grid()
        
    elif widgets["model_inv"].get_value() == "sedov_forward":
        
        widgets["m_eject_inv"].value_var.set("0")
        widgets["m_eject_inv"].input.config(state="disabled")
        widgets["ctau_inv"].input.grid_remove()
        widgets["ctau_inv"].label.grid_remove()
        widgets["s_inv"].input.grid()
        widgets["s_inv"].label.grid()
        widgets["n_inv"].input.grid()
        widgets["n_inv"].label.grid()
        widgets["s_inv"].value_var.set("0")
        widgets["s_inv"].input.config(state="disabled")
        widgets["n_inv"].input.config(state="disabled")
        
        widgets["Te_r_inv"].input.grid_remove()
        widgets["Te_r_inv"].label.grid_remove()
        widgets["EM58_r_inv"].input.grid_remove()
        widgets["EM58_r_inv"].label.grid_remove()
        
        widgets["Te_f_inv"].input.grid()
        widgets["Te_f_inv"].label.grid()
        widgets["EM58_f_inv"].input.grid()
        widgets["EM58_f_inv"].label.grid()
        
    else: # Default standard forward
        widgets["s_inv"].input.config(state="normal")
        widgets["n_inv"].input.config(state="normal")
        widgets["m_eject_inv"].input.config(state="normal")
        widgets["m_eject_inv"].value_var.set("1.2")
        widgets["ctau_inv"].input.grid_remove()
        widgets["ctau_inv"].label.grid_remove()
        widgets["s_inv"].input.grid()
        widgets["s_inv"].label.grid()
        widgets["n_inv"].input.grid()
        widgets["n_inv"].label.grid()
            
        widgets["Te_r_inv"].input.grid_remove()
        widgets["Te_r_inv"].label.grid_remove()
        widgets["EM58_r_inv"].input.grid_remove()
        widgets["EM58_r_inv"].label.grid_remove()
        
        widgets["Te_f_inv"].input.grid()
        widgets["Te_f_inv"].label.grid()
        widgets["EM58_f_inv"].input.grid()
        widgets["EM58_f_inv"].label.grid()
        
        
    if update:
        SNR_INV.update_output()

    
##############################################################################################################################
def s_change_inv(widgets, SNR_INV, update=True):
    """Changes available input parameters when s is changed in the inverse window

    Args:
        update (bool): true if SuperNovaRemnant instance needs to be updated (generally only False when run during
                       initialization to avoid running update_output multiple times unnecessarily)
    """

    if widgets["s_inv"].value_var.get() == '0':
        widgets["model_inv"].input["cloudy_forward"].config(state="normal")
        widgets["model_inv"].input["sedov_forward"].config(state="normal")
        widgets["n_inv"].input.config(values=(6, 7, 8, 9, 10, 11, 12, 13, 14)) 
        
        if widgets["n_inv"].get_value() not in (6, 7, 8, 9, 10, 11, 12, 13, 14):
            # Reset n value if old value not available
            widgets["n_inv"].value_var.set(6)
    else:
        # Restore previous values for input parameters m_w and v_w and reduce available n values
        widgets["model_inv"].input["cloudy_forward"].config(state="disabled")
        widgets["model_inv"].input["sedov_forward"].config(state="disabled")
        widgets["n_inv"].input.config(values=(6, 7, 8, 9, 10, 11, 12, 13, 14))
        
        if widgets["n_inv"].get_value() not in (6, 7, 8, 9, 10, 11, 12, 13, 14):
            # Reset n value if old value not available
            widgets["n_inv"].value_var.set(6)
    if update:
        SNR_INV.update_output()


##############################################################################################################################
def emissivity_window():
    """Create window to display emissivity data for an SNR with input parameters from the main window."""

    window = gui.ScrollWindow()
    window.root.focus()
    if window.os != "Linux":
        window.root.config(cursor="watch")
    else:
        window.root.config(cursor="wait")
    window.root.geometry("%dx%d+%d+%d" %(880, 650, (ws-880)/2, (hs-700)/2))
    window.root.update()
    window.canvas.grid_remove()
    left_frame = gui.LayoutFrame(window.container, 5)
    right_frame = gui.LayoutFrame(window.container, 5, column=1, row=0)
    root_id = str(window.root)
    SNR_EM = calc.SNREmissivity(SNR, root_id)
    gui.InputParam.instances[root_id] = {}
    widgets = gui.InputParam.instances[root_id]
    gui.SectionTitle(right_frame, "Output Plots:")
    energy_frame = gui.LayoutFrame(right_frame, 0, columnspan=2)
    gui.InputEntry(energy_frame, "energy", "Energy for specific intensity plot (keV):", 1,
                   SNR_EM.update_specific_intensity, gt_zero)
    SNR_EM.plots["Inu"] = plt.OutputPlot(gui.LayoutFrame(right_frame, (0, 5)), (5, 2.6),
                                         "Normalized impact parameter",
                                         "Specific intensity/\nerg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$ sr$^{-1}$")
    range_frame = gui.LayoutFrame(right_frame, (0, 5, 0, 0), columnspan=4)
    gui.InputParam(gui.LayoutFrame(range_frame), None, "Energy range (keV):", None)
    gui.InputEntry(gui.LayoutFrame(range_frame, row=0, column=1), "emin", "", 0.3, SNR_EM.update_luminosity_spectrum,
                   lambda value: 0 < value < widgets["emax"].get_value(), padding=(0, 5))
    gui.InputEntry(gui.LayoutFrame(range_frame, row=0, column=3), "emax", "to", 8, SNR_EM.update_luminosity_spectrum,
                   lambda value: value > widgets["emin"].get_value(), padding=(5, 5))
    SNR_EM.plots["Lnu"] = plt.OutputPlot(gui.LayoutFrame(right_frame), (5, 2.6), "Energy/keV",
                                         "Luminosity/\nerg s$^{-1}$ Hz$^{-1}$")
    gui.SectionTitle(left_frame, "SNR Properties:")
    gui.DisplayValue(left_frame, "Age", "yr", SNR.data["t"])
    gui.DisplayValue(left_frame, "Radius", "pc", SNR.calc["r"])
    gui.InputParam(left_frame, label="Model type:  \u200a{}".format(get_model_name(SNR.data["model"], SNR_EM)),
                   padding=(5, 2))
    if SNR_EM.data["model"] == "chev":
        gui.OutputValue(left_frame, "em", "Emission measure:", "cm\u207B\u00B3", 3, padding=(5, 1, 5, 0))
        em_frame = gui.LayoutFrame(left_frame, columnspan=2)
        gui.OutputValue(gui.LayoutFrame(em_frame, column=0, row=0), "em_f", "(Forward:", "cm\u207B\u00B3,", 3,
                        padding=(5, 0, 0, 0), font="-size 9")
        gui.OutputValue(gui.LayoutFrame(em_frame, column=1, row=0), "em_r", "reverse:", "cm\u207B\u00B3)", 3,
                        padding=(5, 0, 0, 0), font="-size 9")
        gui.OutputValue(left_frame, "Tem", "Emission weighted temperature:", "K", 3, padding=(5, 1, 5, 0))
        Tem_frame = gui.LayoutFrame(left_frame, columnspan=2)
        gui.OutputValue(gui.LayoutFrame(Tem_frame, column=0, row=0), "Tem_f", "(Forward:", "K,", 3, padding=(5, 0, 0, 0),
                        font="-size 9")
        gui.OutputValue(gui.LayoutFrame(Tem_frame, column=1, row=0), "Tem_r", "reverse:", "K)", 3,
                        padding=(5, 0, 0, 0), font="-size 9")
    else:
        gui.OutputValue(left_frame, "em", "Emission measure:", "cm\u207B\u00B3", 3)
        gui.OutputValue(left_frame, "Tem", "Emission weighted temperature:", "K", 3)
    gui.SectionTitle(left_frame, "Radial Profiles:", padding=(0, 10, 0, 0))
    SNR_EM.plots["temp"] = plt.OutputPlot(gui.LayoutFrame(left_frame, (0, 5, 0, 10), columnspan=5), (4.5, 2.3),
                                          "Normalized radius", "Temperature/K")
    SNR_EM.plots["density"] = plt.OutputPlot(gui.LayoutFrame(left_frame, 0, columnspan=5), (4.5, 2.3),
                                             "Normalized radius", "Density/g cm$^{-3}$",
                                             sharex=SNR_EM.plots["temp"].graph)
    if SNR_EM.data["model"] == "chev":
        gui.OutputValue(right_frame, "lum", "Luminosity over energy range:", "erg s\u207B\u00B9", 3,
                        padding=(5, 5, 5, 0))
        lum_frame = gui.LayoutFrame(right_frame, columnspan=2)
        gui.OutputValue(gui.LayoutFrame(lum_frame, column=0, row=0), "lum_f", "(Forward:", "erg s\u207B\u00B9,", 3,
                        padding=(5, 0, 0, 0), font="-size 9")
        gui.OutputValue(gui.LayoutFrame(lum_frame, column=1, row=0), "lum_r", "reverse:", "erg s\u207B\u00B9)", 3,
                        padding=(5, 0, 0, 0), font="-size 9")
    else:
        gui.OutputValue(right_frame, "lum", "Luminosity over energy range:", "erg s\u207B\u00B9", 3, padding=(5, 5))
    SNR_EM.plots["Lnu"].properties = {"function": SNR_EM.luminosity_spectrum, "color": "g"}
    SNR_EM.plots["Inu"].properties = {"function": SNR_EM.specific_intensity, "color": "m"}
    SNR_EM.plots["temp"].properties = {"function": lambda x: SNR_EM.vector_temperature(x) * SNR_EM.data["T_s"], "color": "b"}
    SNR_EM.plots["density"].properties = {"function": lambda x: SNR_EM.vector_density(x) * 4 * SNR_EM.data["n_0"] *
                                                                SNR_EM.data["mu_H"] * calc.M_H, "color": "r"}
    SNR_EM.update_output()
    window.canvas.grid()
    window.root.bind("<1>", lambda event: event.widget.focus_set())
    window.root.bind("<Return>", enter_pressed)
    window.root.update()
    window.root.config(cursor="")
    window.canvas.config(scrollregion=(0, 0, window.container.winfo_reqwidth(), window.container.winfo_reqheight()))

##############################################################################################################################
def gt_zero(value):
    """Checks if value is positive and non-zero."""
    
    return value > 0

##############################################################################################################################
if __name__ == '__main__':
    
    ab_window_open = {"ISM": False, "Ejecta": False}
    inverseWindowActive = False
    # Set initial ISM abundance type
    ism_ab_type = "Solar"
    ej_ab_type = "CC"
    APP = gui.ScrollWindow("root")
    root_id = "." + APP.container.winfo_parent().split(".")[1]
    gui.InputParam.instances[root_id] = {}
    widgets = gui.InputParam.instances[root_id]
    SNR = calc.SuperNovaRemnant(root_id)
    SNR.data["abundance"] = ABUNDANCE[ism_ab_type].copy()
    SNR.data["ej_abundance"] = ABUNDANCE[ej_ab_type].copy()
    APP.root.wm_title("SNR Modelling Program")
    if APP.os == "Windows":
        ICON = "data/Crab_Nebula.ico"
        APP.root.tk.call("wm", "iconbitmap", APP.root._w, "-default", ICON)
    ws = APP.root.winfo_screenwidth()
    hs = APP.root.winfo_screenheight()
    if APP.os == "Linux":
        width = 1000
    else:
        width = 930
    APP.root.geometry("%dx%d+%d+%d" %(width, 650, (ws-width)/2, (hs-700)/2))
    APP.root.bind("<1>", lambda event: event.widget.focus_set())
    APP.input = gui.LayoutFrame(APP.container, 2, row=0, column=0)
    gui.SectionTitle(APP.input, "Input parameters:", 2)
  
    # Note time isn't restricted to less than t_mrg - this is accounted for in snr_calc.py
    # If time was restricted, users could become confused due to rounding of displayed t_mrg
    gui.InputEntry(APP.input, "t", "Age (yr):", 100, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "e_51", "Energy (x 10\u2075\u00B9 erg):", 1.0, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "temp_ism", "ISM Temperature (K):", 100, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "m_ej", "Ejected mass (M\u2609):", 1.4, SNR.update_output, gt_zero)
    gui.InputDropdown(APP.input, "n", "Ejecta power-law index, n:", 0, n_change,
                      (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14))
    gui.InputDropdown(APP.input, "s", "CSM power-law index, s:", 0, s_change, (0, 2), state="disabled")
    if APP.os == "Linux":
        ratio_label = "Electron to ion temperature ratio Te/Ti:"
    else:
        ratio_label = "Electron to ion temperature ratio T\u2091\u200a/\u200aT\u1d62\u200a:"
    gui.InputDropdown(APP.input, "T_ratio", ratio_label, "Default", update_ratio, "Custom")
    gui.InputEntry(APP.input, "zeta_m", "Cooling adjustment factor:", 1.0, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "n_0", "ISM number density (cm\u207B\u00B3):", 2.0, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "sigma_v", "ISM turbulence/random speed (km/s):", 7.0, SNR.update_output)
    gui.InputEntry(APP.input, "m_w", "Stellar wind mass loss (M\u2609/yr):", 1e-7, SNR.update_output, gt_zero)
    gui.InputEntry(APP.input, "v_w", "Wind speed (km/s):", 30, SNR.update_output, gt_zero)
    gui.SubmitButton(APP.input, "Change ISM Abundances", lambda: abundance_window(SNR.data["abundance"], "ISM"),
                     sticky="w", padx=5, pady=(0, 5))
    SNR.buttons["ej_ab"] = gui.SubmitButton(
        APP.input, "Change Ejecta Abundances", lambda: abundance_window(SNR.data["ej_abundance"], "Ejecta"), sticky="w",
        padx=5)
    SNR.buttons["inv"] = gui.SubmitButton(APP.input, "Open Inverse Mode", inverse_window, sticky="w", padx=5, pady=(5, 5))
    gui.InputParam(APP.input, label="Model type:")
    
    MODEL_FRAME = gui.LayoutFrame(APP.input, 0, row=100, column=0, columnspan=2)
    gui.InputRadio(MODEL_FRAME, "model", None, "standard", lambda *args: model_change(),
                   (("standard", "Standard"), ("fel", "Fractional energy loss"), ("hld", "Hot low-density media", "\n"),
                    ("cism", "Cloudy ISM"), ("sedtay", "Sedov-Taylor (m\u2091\u2C7C = 0)", "\n")), padding=(10, 0, 0, 0))
    
    gui.InputEntry(APP.input, "gamma_1", "Effective \u03B3\u2081, 1 \u2264 \u03B3\u2081 \u2264 5/3:", 1.666, SNR.update_output,
                   lambda value: 1 <= value <= 5.0 / 3.0)
    
    gui.InputDropdown(APP.input, "c_tau", "C/\u03C4", 2, SNR.update_output, (0, 1, 2, 4))   
                              
    gui.InputEntry(APP.input, "t_fel", "Fractional energy loss model start\ntime, within ST or PDS phase (yr):", 5000,
                   SNR.update_output, lambda value: (SNR.calc["t_st"] <= value or SNR.calc["t_st"] > SNR.calc["t_pds"])
                                                     and value <= min(SNR.calc["t_mrg"]["PDS"], SNR.calc["t_mcs"]))
    
    gui.InputEntry(APP.input, "t_hld", "Hot low-density media\nmodel end time (yr):", "4e5", SNR.update_output,
                   lambda value: SNR.calc["t_c"]*0.1 <= value < 1e9)
    
    SNR.buttons["em"] = gui.SubmitButton(APP.input, "Emissivity", emissivity_window, pady=10)
    
    APP.plot_frame = gui.LayoutFrame(APP.container, (10, 0), row=0, column=1)
    
    gui.SectionTitle(APP.plot_frame, "Output:")
    
    APP.plot_controls = gui.LayoutFrame(APP.plot_frame, 0)
    
    gui.InputRadio(gui.LayoutFrame(APP.plot_controls, 0), "plot_type", "Plot Type:", "r", lambda *args: title_change(),
                   (("r", "Radius"), ("v", "Velocity"), ("eMeas", "Emission Measure", "\n"), ("temper", "Temperature")), padding=(10, 0, 0, 0))
    
    gui.CheckboxGroup(
        gui.LayoutFrame(APP.plot_controls, (80, 0, 0, 0), row=widgets["plot_type"].label.grid_info()["row"], column=1),
        "Log scale:", scale_change, (("x_scale", "x-axis", "0"), ("y_scale", "y-axis", "0")), padding=(10, 0, 0, 0))
    
    plot_container = gui.LayoutFrame(APP.plot_frame, 0)
    
    SNR.graph = plt.TimePlot(plot_container, (6.4, 4.4))
    
    AXIS_FRAME = gui.LayoutFrame(APP.plot_frame, (0, 5), column=0)
    
    gui.InputDropdown(gui.LayoutFrame(AXIS_FRAME, (0, 0, 15, 0), row=0, column=0), "range", "Plotted Time:", "Current",
                      limit_change, ("Current", "Reverse Shock Lifetime", "ED-ST", "PDS", "MCS"), width=19,
                      font="-size 9")
    
    gui.InputSpinbox(gui.LayoutFrame(AXIS_FRAME, 0, row=0, column=1), "xmin", "Min:", "0", increment_xlimits,
                     lambda value: value != widgets["xmax"].get_value(), increment=10, from_=0, to=100000000, width=8)
    
    gui.InputSpinbox(gui.LayoutFrame(AXIS_FRAME, 0, row=0, column=2), "xmax", "Max:", "900", increment_xlimits,
                     lambda value: value != widgets["xmin"].get_value(), increment=100, from_=0, to=100000000, width=8)
    
    APP.output = gui.LayoutFrame(APP.plot_frame, (0, 10), column=0, columnspan=2)
    
    APP.output_values = gui.LayoutFrame(APP.output, 0, row=0, column=0)
    
    APP.output_times = gui.LayoutFrame(APP.output, (20, 0, 0, 0), row=0, column=1)
    
    gui.SectionTitle(APP.output_values, "Values at specified time:", 4, 11)
    gui.SectionTitle(APP.output_times, "Phase transition times:", 4, 11)
    gui.OutputValue(APP.output_values, "T", "Blast-wave shock electron temperature:", "K")
    gui.OutputValue(APP.output_values, "Tr", "Reverse shock electron temperature:", "K")
    gui.OutputValue(APP.output_values, "r", "Blast-wave shock radius:", "pc")
    gui.OutputValue(APP.output_values, "rr", "Reverse shock radius:", "pc")
    gui.OutputValue(APP.output_values, "v", "Blast-wave shock velocity:", "km/s")
    gui.OutputValue(APP.output_values, "vr", "Reverse shock velocity:", "km/s")
    gui.OutputValue(APP.output_values, "epsi", "", "", padding=0)
    gui.OutputValue(APP.output_times, "t-s2", "", "", padding=0)
    gui.OutputValue(APP.output_times, "Core", "", "", padding=0)
    gui.OutputValue(APP.output_times, "Rev", "", "", padding=0)    
    gui.OutputValue(APP.output_times, "t-ST", "", "yr")       #Sedov-Taylor Phase
    gui.OutputValue(APP.output_times, "t-CISM", "", "yr")       #White and Long
    gui.OutputValue(APP.output_times, "t-PDS", "", "yr")      #Pressure Driven Snowplow
    gui.OutputValue(APP.output_times, "t-MCS", "", "yr")      #Momentum Conserving Shell
    gui.OutputValue(APP.output_times, "t-HLD", "", "yr")
    gui.OutputValue(APP.output_times, "t-FEL", "", "yr")
    gui.OutputValue(APP.output_times, "t-MRG", "", "yr")

    APP.root.bind("<Return>", enter_pressed)
    SNR.update_output()
    model_change(False)
    s_change(False)
    n_change(False)
    APP.root.update()
    widgets["t_fel"].value_var.set(round(SNR.calc["t_pds"]))
    APP.canvas.config(scrollregion=(0, 0, APP.container.winfo_reqwidth(), APP.container.winfo_reqheight()))
    APP.root.mainloop()


##############################################################################################################################