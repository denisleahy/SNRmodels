"""SNR Project Calculation Module.

Classes used to represent a general SNR for main window and specific case of SNR for emissivity window.

Authors: Denis Leahy, Bryson Lawton, Jacqueline Williams
Version: Jan 2019
"""

from collections import namedtuple
import numpy as np
import snr_gui as gui
from scipy.optimize import brentq, newton
from scipy.integrate import quad, cumtrapz, nquad
import functools

PC_TO_KM = 3.0857e13  #km
KEV_TO_ERG = 1.6021766208e-9   #erg
BOLTZMANN = 1.380658e-16 #erg/K
PLANCK = 6.62607015e-27 #erg/s
SOLAR_MASS_TO_GRAM = 1.989e33 #g
M_H = 1.673e-24
K_SED = 1.528
PHI_C = 0.5
A_VALS = {"H": 1, "He": 4, "C": 12, "O": 16, "Ne": 20, "N": 14, "Mg": 24, "Si": 28, "Fe": 56, "S": 32}
Z_VALS = {"H": 1, "He": 2, "C": 6, "O": 8, "Ne": 10, "N": 7, "Mg": 12, "Si": 14, "Fe": 26, "S": 16}
XI_0 = 2.026
YR_TO_SEC = 365.25 * 24 * 3600 #s
BETA = 2
PCyr_TO_KMs = PC_TO_KM / YR_TO_SEC # Converts from pc/yr to km/s (9.78e5 km/s)
# _rchg corresponds to values at which behaviour of reverse shock changes (typically t_core, t_rst for n=2)
# Note a_core = 2 * v_core / t_core 
ValueSet = namedtuple("ValueSet", "l_ed phi_ed t_st r_st t_rchg r_rchg v_rchg a_rchg phi_eff f_n alpha")
VALUE_DICT_S0 = {
    0: ValueSet(1.50, 0.3, 0.495, 0.727, 0.495, 0.779, 0.533, 0.106, 0.0961, 3 / (4 * np.pi), 0.6),    
    1: ValueSet(1.10, 0.343, 0.441, 0.703, 0.441, 0.524, 0.635, -0.005, 0.0960, 1 / (2 * np.pi), 0.5),
    2: ValueSet(1.10, 0.343, 0.387, 0.679, 0.387, 0.503, 0.686, -0.115, 0.0947, 1 / (4 * np.pi), 1 / 3),
    4: ValueSet(1.10, 0.343, 0.232, 0.587, 1.2, 0.775, 0.427, 0.712, 0.0791, 0.00645, 0.0746),
    6: ValueSet(1.39, 0.39, 1.04, 1.07, 0.5133, 0.541, 0.527, 0.112, None, None, None),
    7: ValueSet(1.26, 0.47, 0.732, 0.881, 0.3629, 0.469, 0.553, 0.116, None, None, None),
    8: ValueSet(1.21, 0.52, 0.605, 0.788, 0.2922, 0.413, 0.530, 0.139, None, None, None),
    9: ValueSet(1.19, 0.55, 0.523, 0.725, 0.2489, 0.371, 0.497, 0.162, None, None, None),
    10: ValueSet(1.17, 0.57, 0.481, 0.687, 0.2204, 0.340, 0.463, 0.192, None, None, None),
    11: ValueSet(1.16, 0.585, 0.452, 0.661, 0.1987, 0.316, 0.433, 0.222, None, None, None),
    12: ValueSet(1.15, 0.60, 0.424, 0.636, 0.1818, 0.293, 0.403, 0.251, None, None, None),  
    13: ValueSet(1.145, 0.61, 0.406, 0.620, 0.1681, 0.276, 0.378, 0.264, None, None, None),
    14: ValueSet(1.14, 0.62, 0.389, 0.603, 0.1567, 0.259, 0.354, 0.277, None, None, None)
}

#DONT have proper values for everything except L_ed and phi_ed
ValueSet2 = namedtuple("ValueSet2", "l_ed phi_ed a2 bbm")
VALUE_DICT_S2 = {
    0: ValueSet2(1.50, 0.025, 0, 0), 
    1: ValueSet2(1.50, 0.25, 0, 0),
    2: ValueSet2(1.50, 0.25, 0, 0),
    6: ValueSet2(1.4362, 0.2469, 0.5331, 1.3767),
    7: ValueSet2(1.3389, 0.314, 0.2318, 1.2987),
    8: ValueSet2(1.2976, 0.3521, 0.131, 1.2671),
    9: ValueSet2(1.2745, 0.3764, 0.0846, 1.2498),
    10: ValueSet2(1.2596, 0.3929, 0.0593, 1.239),
    11: ValueSet2(1.2492, 0.4051, 0.0439, 1.2314),
    12: ValueSet2(1.2415, 0.4144, 0.0338, 1.2259),  
    13: ValueSet2(1.2356, 0.4221, 0.0268, 1.2217),
    14: ValueSet2(1.231, 0.4283, 0.0218, 1.2184)
}
#Min and max radius value at the start and end of the ChevParker Data files
S0_CHEV_RMIN = {6: 0.90606, 7: 0.93498, 8: 0.95023, 9: 0.95967, 10: 0.96609, 11: 0.97075, 12: 0.97428, 13: 0.97705, 14: 0.97928}
S0_CHEV_RMAX = {6: 1.25542, 7: 1.18099, 8: 1.15397, 9: 1.13993, 10: 1.13133, 11: 1.12554, 12: 1.12136, 13: 1.11822, 14: 1.11576}
S2_CHEV_RMIN = {6: 0.95849, 7: 0.96999, 8: 0.97649, 9: 0.98068, 10: 0.98360, 11: 0.98575, 12: 0.98740, 13: 0.98871, 14: 0.98978}
S2_CHEV_RMAX = {6: 1.37656, 7: 1.29871, 8: 1.26712, 9: 1.24986, 10: 1.23894, 11: 1.23142, 12: 1.22591, 13: 1.22170, 14: 1.21838}

K_DICT = {}
cism_file = open("data/WL91Parameters.csv", "r")
for line in cism_file:
    line = line.rstrip().split(",")
    K_DICT[float(line[0])] = float(line[1]) * 1.528
    
CISM_EM_WEIGHTED = {"beta": {0: 1.2896, 1: 1.370303, 2: 1.369303, 4: 1.08032}, #beta is ratio of em-weighted Tav to Tshock
                  "alpha": {0: 2.6193, 1: 2.476838394791874, 2: 2.637864262949064, 4: 8.0134}} #alpha is ratio of em-weighted n_av to n_0

#Note White&Long solution files are of the form: r/Rshock, P/Pshock, rho/rho_shock, v/Vshock (so vmax is 3/4), column emission measure
lines = np.loadtxt("data/WLsoln0.0_xfgh.txt")
radius2 = lines[:, 0]
pressure2 = lines[:, 1]
density2 = lines[:, 2]


##################################
# Input Data in the following units:
#
#  Age: SNR.data["t"] - (years)
#  Energy: self.data["e_51"] - (x10^51 ergs)
#  ISM Temperature: self.data["temp_ism"] - (K)
#  Ejected Mass: self.data["m_ej"] - (Msun)
#  Electron to Ion Temperature Ratio: self.data["T_ratio"] - (no units)
#  Stellar Wind Mass Loss: self.data["m_w"] - (Msun/yr)
#  Wind Speed: self.data["v_w"] - (km/s)
#


##############################################################################################################################
##############################################################################################################################
class SuperNovaRemnant:
    """Calculate and store data for a supernova remnant.

    Attributes:
        root (str): ID of GUI main window, used to access widgets
        widgets (dict): widgets used for input
        buttons (dict): emissivity and abundance buttons in program window
        data (dict): values from input window
        calc (dict): values calculated from input values
        graph (snr_gui.TimePlot): plot used to show radius and velocity as functions of time
        cnst (dict): named tuple of constants used for supernova remnant calculations, used only for s=0
        radius_functions (dict): functions used to calculate radius as a function of time
        velocity_functions (dict): functions used to calculate velocity as a function of time or radius
        time_functions (dict): functions used to calculate time as a function of radius
    """
###########################################################################################################################
    def __init__(self, root):
        """Initialize SuperNovaRemnant instance.

        Args:
            root (str): ID of parent window, used to access widgets only from parent
        """

        self.root = root
        self.widgets = gui.InputParam.instances[self.root]
        self.buttons = {}
        self.data = {}
        self.calc = {}
        # Graph is defined in snr.py module
        self.graph = None
        # Constants are defined when an n value is specified
        self.cnst = None
        self.radius_functions = {}
        self.velocity_functions = {}
        self.time_functions = {}
        self._init_functions()

############################################################################################################################
    def update_output(self):
        """Recalculate and update data, plot, and output values using input values from main window."""

        self.data.update(gui.InputParam.get_values(self.root))
        self.data["mu_H"] = sum(A_VALS[key] * 10 ** (self.data["abundance"][key] - 12) for key in A_VALS)
        self.data["mu_e"] = self.data["mu_H"] / sum(Z_VALS[key] * 10 ** (self.data["abundance"][key] - 12)
                                                    for key in Z_VALS)
        ab_sum = sum(10 ** (self.data["abundance"][key] - self.data["abundance"]["H"]) for key in A_VALS)
        self.data["mu_I"] = self.data["mu_H"] / ab_sum
        self.data["Z_sq"] = sum(Z_VALS[key] ** 2 * 10 ** (
            self.data["abundance"][key] - self.data["abundance"]["H"]) for key in A_VALS) / ab_sum
        ab_sum_ej = sum(10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS)
        self.data["mu_H_ej"] = sum(
            A_VALS[key] * 10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS)
        self.data["mu_e_ej"] = self.data["mu_H_ej"] / sum(
            Z_VALS[key] * 10 ** (self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in Z_VALS)
        self.data["mu_I_ej"] = self.data["mu_H_ej"] / ab_sum_ej
        self.data["Z_sq_ej"] = sum(Z_VALS[key] ** 2 * 10 ** (
            self.data["ej_abundance"][key] - self.data["ej_abundance"]["H"]) for key in A_VALS) / ab_sum_ej
        # n must be converted to an integer since it is used as a key in the function dictionaries
        self.data["n"] = round(self.data["n"])
        if str(self.widgets["T_ratio"].input.cget("state")) == "readonly" or type(self.data["T_ratio"]) == str:
            # Calculate Te/Ti ratio
            temp_est = (3 * 10 ** 6 * (self.data["t"] / 10 ** 4) ** -1.2 * (self.data["e_51"] / 0.75 / self.data["n_0"]) **
                        0.4)
            func = 5 / 3 * self.data["n_0"] / 81 / temp_est ** 1.5 * self.data["t"] * YR_TO_SEC * np.log(
                1.2 * 10 ** 5 * 0.1 * temp_est * (temp_est / 4 / self.data["n_0"]) ** 0.5)
            ratio = 1 - 0.97 * np.exp(-func ** 0.4 * (1 + 0.3 * func ** 0.6))
            self.widgets["T_ratio"].value_var.set("{:.2g}".format(ratio))
            self.data["T_ratio"] = ratio
        # Calculate phase transition times and all values needed for future radius and velocity calculations.
        if self.data["s"] == 0:
            if "t_pds" in self.calc:
                pds_old = round(self.calc["t_pds"])
            else:
                pds_old = -1

            self.cnst = VALUE_DICT_S0[self.data["n"]]

            self.calc["v_ej"] = (100 * self.data["e_51"] / self.data["m_ej"]) ** 0.5
            self.calc["c_0"] = ((5 * BOLTZMANN * self.data["temp_ism"] / 3 / M_H / self.data["mu_H"]) ** 0.5 /
                                100000)
            self.calc["c_net"] = (self.calc["c_0"] ** 2 + self.data["sigma_v"] ** 2) ** 0.5
            #Characteristic Radius Calculation
            self.calc["r_ch"] = ((self.data["m_ej"] * SOLAR_MASS_TO_GRAM) ** (1 / 3) /
                                 (self.data["n_0"] * self.data["mu_H"] * M_H) ** (1 / 3) / PC_TO_KM / 10 ** 5)
            #Characteristic Time Calculation
            self.calc["t_ch"] = ((self.data["m_ej"] * SOLAR_MASS_TO_GRAM) ** (5 / 6) / (self.data["e_51"] * 10 ** 51) ** 0.5 /
                                 (self.data["n_0"] * self.data["mu_H"] * M_H) ** (1 / 3) / YR_TO_SEC)
            #Characteristic Velocity Calculation
            self.calc["v_ch"] = self.calc["r_ch"] / self.calc["t_ch"] * PCyr_TO_KMs
            #Sedov-Taylor Time Calculation
            self.calc["t_st"] = self.cnst.t_st * self.calc["t_ch"]
            #Pressure Driven Snowplow Time Calculation
            self.calc["t_pds"] = (13300 * self.data["e_51"] ** (3 / 14) * self.data["n_0"] ** (-4 / 7) *
                                  self.data["zeta_m"] ** (-5 / 14))
            #MCS Time Calculation
            if (self.data["model"] == "cism"):
                self.calc["t_mcs"] = ((14.63 * CISM_EM_WEIGHTED["beta"][self.data["c_tau"]] * (self.data["mu_H"] * M_H) ** (3 / 5) 
                / (self.data["zeta_m"] * CISM_EM_WEIGHTED["alpha"][self.data["c_tau"]]) ** (2 / 3) / BOLTZMANN) ** (15 / 28) 
                * (K_DICT[self.data["c_tau"]] * self.data["e_51"] * 10 ** 51 / 4 / np.pi) ** (3 / 14) 
                * self.data["n_0"] ** (-4 / 7) / YR_TO_SEC)
            else:
                self.calc["t_mcs"] = self.calc["t_pds"] * min(61 * self.calc["v_ej"] ** 3 / self.data["zeta_m"] ** (
                    9 / 14) / self.data["n_0"] ** (3 / 7) / self.data["e_51"] ** (3 / 14), 476 / (
                    self.data["zeta_m"] * PHI_C) ** (9 / 14))
            #Reversal Time Calculation
            self.calc["t_rev"] = newton(self.radius_functions[self.data["n"], "late"], 3 * self.calc["t_ch"])
            #Core Time Calculation
            self.calc["t_c"] = ((2 / 5) ** 5 * self.data["e_51"] * 10 ** 51 * XI_0 / (
                self.calc["c_0"] * 100000) ** 5 / (self.data["n_0"] * self.data["mu_H"] * M_H)) ** (
                1 / 3) / YR_TO_SEC
                
            if self.data["model"] == "fel":
                self.calc["gamma_0"] = 5.0/3.0
                self.calc["epsi"] = (4 * (self.calc["gamma_0"] - self.data["gamma_1"])) / ((self.calc["gamma_0"] - 1.0)*((self.data["gamma_1"] + 1)**2))

                # Set default FEL start time to t_PDS if already set to previous t_PDS
            if self.data["t_fel"] == round(pds_old):
                self.widgets["t_fel"].value_var.set(round(self.calc["t_pds"]))
                self.widgets["t_fel"].previous = round(self.calc["t_pds"])
                self.data["t_fel"] = self.calc["t_pds"]
            #Merger Time Calculation
            self.calc["t_mrg"] = self.merger_time(self.data["n"])
            if self.calc["t_mrg"]["ED"] < self.calc["t_st"] and (self.data["model"] == "cism"):
                # Change back to standard model if CISM phase doesn't occur
                self.widgets["model"].value_var.set("standard")
                self.update_output()
                return
            phases = self.get_phases()
            self.calc["t_mrg_final"] = round(self.calc["t_mrg"][phases[-1]])
            
            # Values to be displayed as output, note r and v are added later and rr and vr are updated later if t<t_rev
            output_data = {
                "epsi": "",
                "rr": "N/A",
                "vr": "N/A",
                "Tr": "N/A",
                "Core": "N/A",
                "Rev": "N/A",
                "t-ST": "N/A",
                "t-CISM": self.calc["t_st"],
                "t-PDS": self.calc["t_pds"],
                "t-MCS": self.calc["t_mcs"],
                "t-HLD": self.calc["t_c"] * 0.1,
                "t-FEL": self.data["t_fel"],
                "t-MRG": self.calc["t_mrg_final"]      
            }
            if self.data["model"] != "sedtay":
                output_data["Core"] = "RS Reaches Core:  " + str(round(self.cnst.t_rchg * self.calc["t_ch"],1)) + " yr"
                output_data["Rev"] = "RS Reaches Center:  " + str(round(self.calc["t_rev"],1)) + " yr"
                output_data["t-ST"] = self.calc["t_st"]
            else:
                output_data["Core"] = "RS Reaches Core:  N/A"
                output_data["Rev"] = "RS Reaches Center:  N/A"
                output_data["t-ST"] = "N/A"                
            del output_data["epsi"]
            if self.widgets["model"].get_value() == "fel":
                output_data["epsi"] = "Fractional energy loss \u03B5: " + str(round(self.calc["epsi"],6))
            else:
                output_data["epsi"] = ""
                
            # Check if HLD model is valid with current conditions and change state of radio button accordingly
            if (0.1 * self.calc["t_c"] <= self.calc["t_pds"] and
                    str(self.widgets["model"].input["hld"].cget("state")) == "disabled"):
                self.widgets["model"].input["hld"].config(state="normal")
            elif 0.1 * self.calc["t_c"] > self.calc["t_pds"] and (
                    str(self.widgets["model"].input["hld"].cget("state")) == "normal"):
                self.widgets["model"].input["hld"].config(state="disabled")
                if self.widgets["model"].get_value() == "hld":
                    self.widgets["model"].value_var.set("standard")
                    self.data["model"] = "standard"
                    self.update_output()
                    return
        else: #s=2 case
            output_data = {
                "epsi": "",
                "rr": "N/A",
                "vr": "N/A",
                "Tr": "N/A",
                "Core": "",
                "Rev": "",
                "t-s2": "This model only includes the\nejecta-dominated phase."   # Set transition time output value to display message explaining lack of transition times
            }
            self.cnst = VALUE_DICT_S2[self.data["n"]]
            # Change units of m_w and v_w to fit with those used in Truelove and McKee
            if (self.data["n"] > 5):
                mdot_GRAMsec = self.data["m_w"]*SOLAR_MASS_TO_GRAM/YR_TO_SEC
                mej_GRAM = self.data["m_ej"]*SOLAR_MASS_TO_GRAM
                self.calc["q"] = (mdot_GRAMsec/10**5)/(4*np.pi*self.data["v_w"]) #
                self.calc["gcn"] = ((1-(3/self.data["n"]))*mej_GRAM) / (((4*np.pi)/3) * ((10/3)*((self.data["n"]-5)/(self.data["n"]-3))*((self.data["e_51"] * 10**51)/(mej_GRAM)))**((3-self.data["n"])/2))
                self.calc["RCn"] = ((self.cnst.a2*self.calc["gcn"])/(self.calc["q"]))**(1/(self.data["n"]-2)) #cm
                self.calc["v0"] = (((self.data["e_51"]*10**51*10*(self.data["n"]-5))/(mej_GRAM*3*(self.data["n"]-3)))**0.5)/(10**5) #km/s
            # Change units of m_w and v_w to fit with those used in Truelove and McKee
            self.data["m_w"] /= 1e-5  #units [(Msun*10^5)/yr]
            self.data["v_w"] /= 10    #units [(km*10^2)/s]
            self.calc["r_ch"] = 12.9 * self.data["m_ej"] / self.data["m_w"] * (self.data["v_w"]) #pc
            self.calc["t_ch"] = (1770 * self.data["e_51"] ** -0.5 * self.data["m_ej"] ** 1.5 / self.data["m_w"] * self.data["v_w"]) #yrs
            self.calc["v_ch"] = self.calc["r_ch"] / self.calc["t_ch"] * PCyr_TO_KMs  #km/s

            # Note despite the key "t_mrg", this is simply the model ending time since only the ED phase is used
            self.calc["t_mrg"] = {"s2": self.calc["t_ch"]}
            phases = self.get_phases()
            self.calc["t_mrg_final"] = round(self.calc["t_mrg"][phases[-1]])
        # Get correct merge time to set spinbox maximum values
        if self.data["model"] == "fel" and self.data["t_fel"] > self.calc["t_mrg_final"]:
            t_fel = round(self.calc["t_st"])
            self.widgets["t_fel"].value_var.set(t_fel)
            self.data["t_fel"] = t_fel
            self.widgets["t_fel"].previous = t_fel
            output_data["t-MRG-FEL"] = self.merger_time(self.data["n"])["FEL"]
            phases = self.get_phases()
        if self.data["t"] >= self.calc["t_mrg_final"]:
            self.data["t"] = self.calc["t_mrg_final"]
            self.widgets["t"].value_var.set(self.calc["t_mrg_final"])
        output_data.update(self.get_specific_data())
        self.calc["r"] = output_data["r"]
        self.calc["T"] = output_data["T"]  #This T is electron shock temperature
        self.widgets["xmin"].input.config(to=self.calc["t_mrg_final"])
        self.widgets["xmax"].input.config(to=self.calc["t_mrg_final"])
        self.update_plot(phases)
        if (self.widgets["model"].get_value() == "cism" and self.widgets["c_tau"].get_value() == 0):
            gui.OutputValue.update(output_data, self.root, 1, phases)
        else:
            gui.OutputValue.update(output_data, self.root, 0, phases)
        if self.emissivity_model_exists():
            self.buttons["em"].config(state="normal")
        else:
            self.buttons["em"].config(state="disabled")
            
###########################################################################################################################
    def emissivity_model_exists(self):
        """Check if an emissivity model is available for the current set of input parameters.

        Returns:
            bool: True if the model exists, False otherwise
        """

        time = self.data["t"]
        if (self.data["n"] in (6,7,8,9,10,11,12,13,14)):
            if (((self.data["model"] in ("standard")) and (self.data["s"] == 2)) or 
                ((self.data["model"] in ("standard")) and (self.data["s"] == 0) and (time < (self.cnst.t_rchg * self.calc["t_ch"]))) or 
                ((self.data["model"] in ("cism")) and (self.data["s"] == 0) and (self.calc["t_rev"] < time < min(self.calc["t_mcs"], self.calc["t_mrg_final"])))):
                return True
        elif (self.data["n"] in (0,1,2,4)):
            if (((self.data["model"] in ("cism")) and (self.data["s"] == 0) and (self.calc["t_rev"] < time < min(self.calc["t_mcs"], self.calc["t_mrg_final"]))) or
                ((self.data["model"] in ("sedtay")) and (self.data["s"] == 0) and (time < min(self.calc["t_pds"], self.calc["t_mrg_final"])))):
                return True
        else:
            return False
        
##########################################################################################################################
    def update_plot(self, phases):
        """Add necessary lines to output plot and redraw with the correct limits.

        Args:
            phases (list): list of SNR phases for current model
        """

        plot_data = self.get_plot_data(phases)
        # Clear previous lines
        self.graph.clear_plot()
        # Add data for forward and reverse shocks - radius vs. time or velocity vs. time depending on plot_type
        direction = "forward"
        self.graph.add_data(plot_data[direction]["t"], plot_data[direction][self.data["plot_type"]],
                            color="r", label="Blast-Wave Shock")
        if self.data["model"] != "sedtay":
            direction = "reverse"
            self.graph.add_data(plot_data[direction]["t"], plot_data[direction][self.data["plot_type"]],
                            color="b", label="Reverse Shock")
            
        # Add vertical lines to show phase transitions
        if self.data["model"] != "sedtay": 
            if "ST" in phases or "CISM" in phases:
                self.graph.graph.axvline(x=self.calc["t_st"], ls="dashed", c="black", label=r"$t_{\mathrm{ST}}$")
            if "PDS" in phases:
                self.graph.graph.axvline(x=self.calc["t_pds"], ls="-.", c="black", label=r"$t_{\mathrm{PDS}}$")
            if "MCS" in phases:
                self.graph.graph.axvline(x=self.calc["t_mcs"], ls="dotted", c="black", label=r"$t_{\mathrm{MCS}}$")
            if "FEL" in phases:
                self.graph.graph.axvline(x=self.data["t_fel"], ls="dotted", lw=3, c="black", label=r"$t_{\mathrm{FEL}}$")
            if "HLD" in phases:
                self.graph.graph.axvline(x=self.calc["t_c"]*0.1, ls="dotted", lw=3, c="black", label=r"$t_{\mathrm{HLD}}$")
        # Update plot display
        self.graph.display_plot(self.get_limits())

############################################################################################################################
    def get_limits(self):
        """Get limits for x-axis on output plot.

        Returns:
            list: minimum and maximum values for x-axis on radius/velocity vs. time plot
        """

        # Account for FEL and HLD models ending other phases earlier than expected
        if self.data["model"] == "fel":
            alt_upper = self.data["t_fel"]
        elif self.data["model"] == "hld":
            alt_upper = self.calc["t_c"] * 0.1
        else:
            # Ensure that alt_upper will never be the minimum value in expressions below
            if self.data["s"] != 2 and self.calc["t_mrg"]["ED"] < self.calc["t_st"]:
                alt_upper = self.calc["t_mrg"]["ED"]
            else:
                alt_upper = float("inf")
        if "Current" in self.data["range"]:
            # Detect current phase and switch range to that type
            phase = self.get_phase(self.data["t"])
            if phase in ("ED", "ST") and "ED-ST" in self.widgets["range"].input.cget("values"):
                phase = "ED-ST"
            elif phase in ("ED", "CISM") and "ED-CISM" in self.widgets["range"].input.cget("values"):
                phase = "ED-CISM"
            self.data["range"] = phase
            # Change s2 to ED for display purposes - s2 represents the ED phase for the case s=2
            if phase == "s2":
                phase = "ED"
            self.widgets["range"].value_var.set("Current ({})".format(phase))
        if self.data["range"] in ("ED-ST", "ED", "ST"):
            limits = (0, min(self.calc["t_pds"], self.calc["t_mrg"]["ST"], alt_upper))
        elif self.data["range"] == "PDS":
            limits = (self.calc["t_pds"], min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"], alt_upper))
        elif self.data["range"] == "MCS":
            limits = (self.calc["t_mcs"], self.calc["t_mrg"]["MCS"])
        elif self.data["range"] == "Custom":
            limits = (self.data["xmin"], self.data["xmax"])
        elif self.data["range"] == "FEL":
            limits = (self.data["t_fel"], self.calc["t_mrg"]["FEL"])
        elif self.data["range"] == "HLD":
            limits = (self.calc["t_c"] * 0.1, self.calc["t_mrg"]["HLD"])
        elif self.data["range"] == "s2":
            limits = (0, self.calc["t_ch"])
        elif self.data["range"] in ("CISM", "ED-CISM"):
            limits = (0, min(self.calc["t_mrg"]["CISM"], self.calc["t_mcs"], alt_upper))
        else:
            # Corresponds to reverse shock lifetime
            limits = (0, min(self.calc["t_rev"], self.calc["t_mrg_final"]))
        if self.data["range"] != "Custom":
            # Set spinbox limit values to match new x-axis limits
            self.widgets["xmin"].value_var.set(int(round(limits[0])))
            self.widgets["xmax"].value_var.set(int(round(limits[1])))
            self.data["xmin"] = limits[0]
            self.data["xmax"] = limits[1]
        else:
            # Ensure range does not include values greater than the model end time
            maximum = self.widgets["xmax"].input.cget("to")
            if limits[0] > maximum and limits[1] > maximum:
                # Prevents code below from making both upper and lower limits have the same value
                limits = (0, maximum)
                self.widgets["xmin"].value_var.set(round(limits[0]))
                self.widgets["xmax"].value_var.set(round(limits[1]))
                self.data["xmin"] = limits[0]
                self.data["xmax"] = limits[1]
            elif limits[1] > maximum:
                limits = (limits[0], maximum)
                self.widgets["xmax"].value_var.set(round(limits[1]))
                self.data["xmax"] = limits[1]
            elif limits[0] > maximum:
                limits = (maximum, limits[0])
                self.widgets["xmin"].value_var.set(round(limits[0]))
                self.data["xmin"] = limits[0]
        return limits

############################################################################################################################
    def get_phase(self, time):
        """Get current phase of SNR model.

        Args:
            time (float): age of SNR

        Returns:
            str: abbreviation representing phase of model
        """

        if self.data["s"] == 2:
            phase = "s2"
        elif self.data["model"] == "fel" and time > self.data["t_fel"]:
            phase = "FEL"
        elif self.data["model"] == "hld" and time > self.calc["t_c"] * 0.1:
            phase = "HLD"
        elif (self.data["model"] == "cism") and time > self.calc["t_st"]:
            if time > self.calc["t_mcs"]:
                phase = "MCS"
            else:
                phase = "CISM"
        elif (time < self.calc["t_st"] < self.calc["t_pds"]) or (time < self.calc["t_pds"] < self.calc["t_st"]):
            phase = "ED"
        elif time < self.calc["t_pds"]:
            phase = "ST"
        elif time < self.calc["t_mcs"]:
            phase = "PDS"
        else:
            phase = "MCS"
        return phase

###########################################################################################################################
    def get_plot_data(self, phases):
        """Get of forward and reverse shock data used to plot radius vs. time or velocity vs. time

        Args:
            phases (list): list of SNR phases for current model

        Returns:
            dict: dictionary of np.ndarrays with shock radius, velocity, and time used to create output plots
        """

        plot_data = {}
        # Account for different reverse shock phases between s=0 and s=2
        if self.data["s"] == 0:
            rev_phases = ("early", "late")
        else:
            rev_phases = "s2r",
        all_phases = {"forward": phases, "reverse": rev_phases}
        for direction, phase_list in all_phases.items():
            plot_data[direction] = {"t": [], "r": [], "v": []}
            for phase in phase_list:
                t, r, v = self.get_data(phase)
                plot_data[direction]["t"] = np.concatenate([plot_data[direction]["t"], t])
                plot_data[direction]["r"] = np.concatenate([plot_data[direction]["r"], r])
                plot_data[direction]["v"] = np.concatenate([plot_data[direction]["v"], v])
        return plot_data

###########################################################################################################################
    def get_specific_data(self):
        """Get output values (radius, velocity, and temperature of both shocks) at a specific time.

        Returns:
            dict: dictionary of output values
        """

        output = {}
        time = self.data["t"]
        phase = self.get_phase(time)
        t, output["r"], output["v"] = self.get_data(phase, time)
        if self.data["model"] != "sedtay":
            if self.data["s"] == 2:
                t, output["rr"], output["vr"] = self.get_data("s2r", time)
            elif time < self.cnst.t_rchg * self.calc["t_ch"]:
                t, output["rr"], output["vr"] = self.get_data("early", time)
            elif time < self.calc["t_rev"]:
                t, output["rr"], output["vr"] = self.get_data("late", time)
        output["T"] = (self.data["T_ratio"]*(3 / 16 * self.data["mu_I"] * M_H / BOLTZMANN * (output["v"] * 100000) ** 2))  #T_e_shock
        if "vr" in output:
            output["Tr"] = (self.data["T_ratio"]*(3 / 16 * self.data["mu_I"] * M_H / BOLTZMANN * (output["vr"] * 100000) ** 2)) #T_e_shock
        return output

#############################################################################################################################
    def get_data(self, phase, t=None):
        """Returns dictionary of time, radius, and velocity data for a forward or reverse shock.

        Args:
            phase (str): phase to get data for
            t (float, optional): time, only specified for finding values at a specific time rather than a list of times

        Returns:
            np.ndarray/float: time value(s)
            np.ndarray/float: radius value(s)
            np.ndarray/float: velocity value(s)
            Note that float values are returned if t is specified, np.ndarray objects are returned otherwise.
        """
        output_values = self.velocity(phase, self.radius_time(phase, t))
        return output_values["t"], output_values["r"], output_values["v"]

############################################################################################################################
    def time_array(self, phase):
        """Returns an array of times appropriate for a given phase.

        Args:
            phase (str): phase to get data for

        Returns:
            np.ndarray: array of time values used to create plots for given phase
        """

        # Account for phases that may end early due to start of FEL or HLD phases
        if self.data["model"] == "fel":
            alt_upper = self.data["t_fel"]
        elif self.data["model"] == "hld":
            alt_upper = self.calc["t_c"] * 0.1
        else:
            # Ensure alt_upper won't be found as a minimum in the expressions below
            alt_upper = float("inf")
        if phase in ("s2", "s2r"):
            t_array = np.concatenate([np.linspace(1, 100, 1000), np.linspace(100, self.calc["t_ch"], 50000)])
        elif phase == "ED":
            t_array = np.linspace(1, min(self.calc["t_st"], alt_upper, self.calc["t_pds"], self.calc["t_mrg"]["ED"]),
                                  1000)
        elif phase == "ST":
            t_array = np.linspace(self.calc["t_st"], min(self.calc["t_pds"], self.calc["t_mrg"]["ST"], alt_upper), 1000)
        elif phase == "PDS":
            t_array = np.linspace(self.calc["t_pds"], min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"], alt_upper),
                                  10000)
        elif phase == "MCS":
            t_array = np.linspace(self.calc["t_mcs"], self.calc["t_mrg"]["MCS"], 10000)
        elif phase == "FEL":
            t_array = np.linspace(self.data["t_fel"], self.calc["t_mrg"]["FEL"], 50000)
        elif phase == "HLD":
            t_array = np.linspace(self.calc["t_c"] * 0.1, self.calc["t_mrg"]["HLD"],
                                  int(round(self.calc["t_mrg"]["HLD"]/10)))
        elif phase == "CISM":
            t_array = np.linspace(self.calc["t_st"], min(self.calc["t_mrg"]["CISM"], self.calc["t_mcs"]), 50000)
        elif phase == "early":
            t_array = np.linspace(1, self.cnst.t_rchg * self.calc["t_ch"], 1000)
        elif phase == "late":
            t_array = np.linspace(self.cnst.t_rchg * self.calc["t_ch"], self.calc["t_rev"], 1000)
        else:
            t_array = []
        return t_array

##########################################################################################################################
    def radius_time(self, phase, t=None):
        """Returns a dictionary with time and radius data.

        Args:
            phase (str): phase to get data for
            t (float, optional): time, only specified for finding values at a specific time rather than a list of times

        Returns:
            dict: dictionary with radius and time values
        """

        if t is None:
            # Generate arrays of data for plotting purposes

            # Determine input type and function needed to compute results
            if (self.data["n"] in (1,2, 4) and (phase == "ED" or phase == "early")) or (                              #Fix this
                    self.data["n"] < 3 and phase in ("s2", "s2r")):
                # Set input as radius and use a time function since these cases provided t(r) rather than r(t)
                if phase == "ED" and self.data["s"] == 0:
                    r_chg = self.cnst.r_st * self.calc["r_ch"]
                    num = 100
                elif phase == "s2":
                    r_chg = 1 / 1.5 ** 2 / (3 - self.data["n"]) * self.calc["r_ch"] - 1
                    num = 100000
                elif phase == "s2r":
                    r_chg = 1 / 1.5 ** 2 / (3 - self.data["n"]) / 1.19 * self.calc["r_ch"] - 1
                    num = 100000
                else:
                    r_chg = self.cnst.r_rchg * self.calc["r_ch"]
                    num = 100
                # Create array of radius values
                input_arr = np.linspace(0.01, r_chg, num)
                input_key = "r"
                output_key = "t"
                output_func = self.time_functions[self.data["n"], phase]
            else:
                # Use time as the input and a function to determine radius these cases provided r(t)
                input_arr = self.time_array(phase)
                input_key = "t"
                output_key = "r"
                output_func = self.radius_functions[self.data["n"], phase]
            # Generate output
            output = {
                input_key: input_arr,
                output_key: output_func(np.array(input_arr))
            }
        else:
            # Find output values only at a specific time
            output = {
                "r": self.radius_functions[self.data["n"], phase](t),
                "t": t
            }
        return output

##########################################################################################################################
    def velocity(self, phase, output):
        """Returns dictionary of output with velocity added.

        Args:
            phase (str): phase to get data for
            output (dict): partially completed output dictionary, output of radius_time

        Returns:
            dict: dictionary including radius, time, and velocity (radius and time from output argument dictionary)
        """

        # Determine whether the input needs to be radius or time depending on if t(r) or r(t) is used
        if (self.data["n"] in (1, 2, 4) and ((phase in ("ED", "early")) or (self.data["s"] == 2))):
            input_ = output["r"]
        else:
            input_ = output["t"]
        if isinstance(input_, np.ndarray):
            # Ensure that original input arrays are not altered in the velocity functions
            input_ = np.array(input_)
        output["v"] = self.velocity_functions[self.data["n"], phase](input_)
        return output

#########################################################################################################################
    def _s0n0_solution(self, key, phase):
        """Returns specific type of function used to calculate values for the s=0, n=0 case.

        Args:
            key (str): determines which function is returned, see function_dict for possible values
            phase (str): evolutionary phase of SNR to get data for

        Returns:
            function: a function to calculate the parameter specified for s=0, n=0 case
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            if phase == "ED":
                return 2.01 * t * (1 + 1.72 * t ** 1.5) ** (-2 / 3) * self.calc["r_ch"]
            else:
                return (1.42 * t - 0.254) ** 0.4 * self.calc["r_ch"]

        def radius_r(t):
            t /= self.calc["t_ch"]
            if phase == "early":
                return 1.83 * t * (1 + 3.26 * t ** 1.5) ** (-2 / 3) * self.calc["r_ch"]
            else:
                return t * (0.779 - 0.106 * t - 0.533 * np.log(t)) * self.calc["r_ch"]

        def velocity_b(t):
            t /= self.calc["t_ch"]
            if phase == "ED":
                return 2.01 * (1 + 1.72 * t ** 1.5) ** (-5 / 3) * self.calc["v_ch"]
            else:
                return 0.569 * (1.42 * t - 0.254) ** -0.6 * self.calc["v_ch"]

        def velocity_r(t):
            t /= self.calc["t_ch"]
            if phase == "early":
                return 5.94 * t ** 1.5 * (1 + 3.26 * t ** 1.5) ** (-5 / 3) * self.calc["v_ch"]
            else:
                return (0.533 + 0.106 * t) * self.calc["v_ch"]

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
  
################################################################################################################
    def _s2nlt3_solution(self, n, key):
        """Returns a specific type of function used for the s=2, n<3 case.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for s=2, n<3 case
        """

        def time_b(r):
            r /= self.calc["r_ch"]
            return (0.594 * ((3 - n) / (5 - n)) ** 0.5 * r * (1 - 1.5 * (3 - n) ** 0.5 * r ** 0.5) ** (-2 / (3 - n)) *
                    self.calc["t_ch"])

        def time_r(r):
            l_ed = 1.19
            return time_b(r * l_ed)

        def velocity_b(r):
            r /= self.calc["r_ch"]
            return 1.68 * ((5 - n) / (3 - n)) ** 0.5 * (1 - 1.5 * (3 - n) ** 0.5 * r ** 0.5) / (1 + 1.5 * (n - 2) / (
                3 - n) ** 0.5 * r ** 0.5) * self.calc["v_ch"]

        def velocity_r(r):
            r /= self.calc["r_ch"]
            return (2.31 * (5 - n) ** 0.5 / (3 - n) * r ** 0.5 * (1 - 1.63 * (3 - n) ** 0.5 * r ** 0.5) / (1 + 1.63 * (n - 2) / (3 - n) ** 0.5 * r ** 0.5) * self.calc["v_ch"])

        function_dict = {
            "t": time_b,
            "tr": time_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
########################################################################################################################
    def _s2ngt5_solution(self, n, key):
        """Returns a specific type of function used for the s=2, n>5 cases.

        Args:
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for s=2, n>5 cases
        """
        #Units cm/s
        def radius_b(t):
            return (self.calc["RCn"]/(PC_TO_KM*10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))  

        def radius_r(t):
            return (self.calc["RCn"]/(PC_TO_KM*10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))/self.cnst.l_ed

        def velocity_b(t):
            return ((n-3)/(n-2)) *((self.calc["RCn"]/(10**5))*self.cnst.bbm) * (t*YR_TO_SEC) ** ((-1)/(n-2)) 

        def velocity_r(t):
            vf2 = ((n-3)/(n-2)) *((self.calc["RCn"]/(10**5))*self.cnst.bbm) * (t*YR_TO_SEC) ** ((-1)/(n-2))
            rr2 = (self.calc["RCn"]/(10**5))*self.cnst.bbm * (t*YR_TO_SEC) ** ((n-3)/(n-2))/self.cnst.l_ed
            return (rr2/(t*YR_TO_SEC)) - (vf2/self.cnst.l_ed)
        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
#########################################################################################################################
    def _ed_solution(self, n, key):
        """General ED solution as defined in Truelove and McKee.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the ED solution
        """

        def time_b(r):
            r /= self.calc["r_ch"]
            return ((self.cnst.alpha / 2) ** 0.5 * r / self.cnst.l_ed * (1 - (3 - n) / 3 * (
                self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * r ** 1.5) ** (
                -2 / (3 - n)) * self.calc["t_ch"])

        def time_r(r):
            r /= self.calc["r_ch"]
            return (self.cnst.alpha / 2) ** 0.5 * r * (1 - (3 - n) / 3 * (
                self.cnst.phi_ed / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * (r * self.cnst.l_ed) ** 1.5) ** ( -2 / (
                3 - n)) * self.calc["t_ch"]

        def velocity_b(r):
            r /= self.calc["r_ch"]
            return ((2 / self.cnst.alpha) ** 0.5 * self.cnst.l_ed * ((1 - (3 - n) / 3 * (
                self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n) ** 0.5 * r ** 1.5) ** ((5 - n) / (3 - n))) / (
                1 + n / 3 * (self.cnst.phi_eff / self.cnst.l_ed / self.cnst.f_n)** 0.5 * r ** 1.5) * self.calc["v_ch"])

        def velocity_r(r):
            r /= self.calc["r_ch"]
            return ((2 * self.cnst.phi_ed / self.cnst.alpha / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5 * (
                (1 - (3 - n) / 3 * (self.cnst.phi_ed / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5) ** (
                2 / (3 - n))) / (1 + n / 3 * (self.cnst.phi_ed / self.cnst.f_n) ** 0.5 * self.cnst.l_ed * r ** 1.5) *
                    self.calc["v_ch"])

        function_dict = {
            "t": time_b,
            "tr": time_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]
    
########################################################################################################################
    def _st_solution(self, key):
        """Offset ST solution for forward shock and constant acceleration solution for reverse shock.

        Args:
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the ST solution
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            return ((self.cnst.r_st ** 2.5 + XI_0 ** 0.5 * (t - self.cnst.t_st)) ** 0.4 *
                    self.calc["r_ch"])

        def velocity_b(t):
            t /= self.calc["t_ch"]
            return (2 / 5 * XI_0 ** 0.5 * (self.cnst.r_st ** 2.5 + XI_0 ** 0.5 * (t - self.cnst.t_st)) **
                    -0.6 * self.calc["v_ch"])

        def radius_r(t):
            t /= self.calc["t_ch"]
            return (t * (self.cnst.r_rchg / self.cnst.t_rchg - self.cnst.a_rchg * (t - self.cnst.t_rchg) - (
                self.cnst.v_rchg - self.cnst.a_rchg * self.cnst.t_rchg) * np.log(t / self.cnst.t_rchg)) *
                    self.calc["r_ch"])

        def velocity_r(t):
            t /= self.calc["t_ch"]
            return (self.cnst.v_rchg + self.cnst.a_rchg * (t - self.cnst.t_rchg)) * self.calc["v_ch"]

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]

####################################################################################################################
    def _cn_solution(self, n, key):
        """CN solution from Truelove and Mckee.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified using the CN solution
        """

        def radius_b(t):
            t /= self.calc["t_ch"]
            return (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (10 / 3 * (n - 5) / (
                n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** ((n - 3) / n) * self.calc["r_ch"]

        def radius_r(t):
            return radius_b(t) / self.cnst.l_ed

        def velocity_b(t):
            t /= self.calc["t_ch"]
            return ((n - 3) / n * (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (
                10 / 3 * (n - 5) / (n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** (-3 / n) * self.calc["v_ch"])

        def velocity_r(t):
            t /= self.calc["t_ch"]
            return (3 / n / self.cnst.l_ed * (27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (
                n - 3) / self.cnst.phi_ed * (10 / 3 * (n - 5) / (n - 3)) ** ((n - 3) / 2)) ** (1 / n) * t ** (
                -3 / n) * self.calc["v_ch"])

        function_dict = {
            "r": radius_b,
            "rr": radius_r,
            "v": velocity_b,
            "vr": velocity_r
        }
        return function_dict[key]

########################################################################################################################
    def _pds_solution(self, n, key):
        """PDS solution adapted from Cioffi et al.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for PDS phase
        """

        def radius(t):
            if self.calc["t_pds"] < self.calc["t_st"]:
                previous = "ED"
            else:
                previous = "ST"
            r_end = self.radius_functions[n, previous](self.calc["t_pds"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_pds"], t)[0]
            return r

        def velocity(t):
            t /= self.calc["t_pds"]
            if isinstance(t, float):
                if t < 1.1:
                    return lin_velocity(t)
                else:
                    return reg_velocity(t)
            else:
                return np.concatenate([lin_velocity(t[np.where(t < 1.1)]), reg_velocity(t[np.where(t >= 1.1)])])

        def reg_velocity(t):
            v_pds = 413 * self.data["n_0"] ** (1 / 7) * self.data["zeta_m"] ** (3 / 14) * self.data["e_51"] ** (1 / 14)
            return v_pds * (4 * t / 3 - 1 / 3) ** -0.7

        def lin_velocity(t):
            """Linear velocity function to join previous phase to v_pds (reg_velocity)."""

            if self.calc["t_pds"] < self.calc["t_st"]:
                previous = "ED"
            else:
                previous = "ST"
            v_end = self.velocity_functions[n, previous](self.calc["t_pds"])
            return (reg_velocity(1.1) - v_end) / 0.1 * (t - 1) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

######################################################################################################################
    def _mcs_solution(self, n, key):
        """MCS solution adapted from Cioffi et al.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for MCS phase
        """

        def radius(t):
            if (self.data["model"] == "cism"):
                previous = "CISM"
            else:
                previous = "PDS"
            r_end = self.radius_functions[n, previous](self.calc["t_mcs"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_mcs"], t)[0]
            return r

        def velocity(t):
            if (self.data["model"] == "cism"):
                v_mcs = self.velocity_functions[n, "CISM"](self.calc["t_mcs"])
                r_mcs = self.radius_functions[n, "CISM"](self.calc["t_mcs"])
                return v_mcs * (1 + 4 * v_mcs / r_mcs * (t - self.calc["t_mcs"]) / PCyr_TO_KMs) ** (-3 / 4)
            else:
                t /= self.calc["t_pds"]
                t_ratio = self.calc["t_mcs"] / self.calc["t_pds"]
                if isinstance(t, float):
                    if t < t_ratio * 1.1:
                        return lin_velocity(t)
                    else:
                        return reg_velocity(t)
                else:
                    return np.concatenate([lin_velocity(t[np.where(t < t_ratio * 1.1)]),
                                           reg_velocity(t[np.where(t >= t_ratio * 1.1)])])

        def reg_velocity(t):
            r_pds = 14 * self.data["e_51"] ** (2 / 7) / self.data["n_0"] ** (3 / 7) / self.data["zeta_m"] ** (1 / 7)
            r_mcs = (4.66 * t * (1 - 0.939 * t ** -0.17 + 0.153 / t)) ** 0.25 * r_pds
            t_mcs = self.calc["t_mcs"] / self.calc["t_pds"]
            return PCyr_TO_KMs * r_pds / 4 * (4.66 / self.calc["t_pds"] * (1 - 0.779 * t_mcs ** -0.17)) * (
                4.66 * (t - t_mcs) * (1 - 0.779 * t_mcs ** -0.17) + (r_mcs / r_pds) ** 4) ** -0.75

        def lin_velocity(t):
            """Linear velocity function to join previous phase to v_mcs (reg_velocity)."""

            v_end = self.velocity_functions[n, "PDS"](self.calc["t_mcs"])
            t_ratio = self.calc["t_mcs"] / self.calc["t_pds"]
            return (reg_velocity(t_ratio * 1.1) - v_end) / 0.1 / t_ratio * (t - t_ratio) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

##########################################################################################################################
    def _fel_solution(self, n, key):
        """Fractional energy loss solution from Liang and Keilty.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for FEL model
        """

        def radius(t):
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"])
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"]) / PCyr_TO_KMs
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4 #LK Equation 8
            n1 = 1 / (4 - 3 * alpha_1) #LK Equation Under 5
            return (r_0 ** (1 / n1) + (4 - 3 * alpha_1) * v_0 * (t - self.data["t_fel"]) / r_0 ** (3 * (
                alpha_1 - 1))) ** n1

        def velocity(t):
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"])
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"])
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4 #LK Equation 8
            return v_0 * (radius(t) / r_0) ** (3 * (alpha_1 - 1))

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]

#########################################################################################################################
    def _hld_solution(self, n, key):
        """High temperature solution adapted from Tang and Wang.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for HLD model
        """

        def velocity(t):
            t_0 = 0.1 * self.calc["t_c"]
            phase = self.get_phase(t_0)
            correction = ((self.velocity_functions[n, phase](t_0) / self.calc["c_0"]) ** (5 / 3) - 1) / 10
            return self.calc["c_0"] * (correction * self.calc["t_c"] / t + 1) ** (3 / 5)

        def radius(t):
            t_0 = 0.1 * self.calc["t_c"]
            phase = self.get_phase(t_0)
            r_0 = self.radius_functions[n, phase](t_0)
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(t)/PCyr_TO_KMs, t, initial=0) + r_0
            else:
                correction = ((self.velocity_functions[n, phase](t_0) / self.calc["c_0"]) ** (5 / 3) - 1) / 10
                r = r_0 + self.calc["c_0"] * self.calc["t_c"] / PCyr_TO_KMs * quad(lambda t_dl: (
                    correction / t_dl + 1) ** (3 / 5), 0.1, t / self.calc["t_c"])[0]
            return r

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]
##########################################################################################################################
    def _cism_solution(self, n, key):
        """Cloudy ISM solution from White and Long.

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for CISM model
        """

        def radius(t):
            r_end = self.radius_functions[n, "ED"](self.calc["t_st"])
            if isinstance(t, np.ndarray):
                r = cumtrapz(velocity(np.array(t))/PCyr_TO_KMs, t, initial=0) + r_end
            else:
                r = r_end + quad(lambda t: velocity(t)/PCyr_TO_KMs, self.calc["t_st"], t)[0]
            return r

        def velocity(t):
            transition_end = self.calc["t_st"] * 1.1
            if isinstance(t, np.ndarray):
                return np.concatenate([lin_velocity(t[np.where(t < transition_end)]),
                                       reg_velocity(t[np.where(t >= transition_end)])])
            else:
                if t < transition_end:
                    return lin_velocity(t)
                else:
                    return reg_velocity(t)

        def reg_velocity(t):
            k_cism = K_DICT[self.data["c_tau"]]
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            gamma = 5 / 3
            r_cism = (25 * (gamma + 1) * k_cism * self.data["e_51"] * 10 ** 51 / 16 / np.pi / rho_0) ** 0.2 * (
                t * YR_TO_SEC) ** 0.4 #r_s in White and Long
            return ((gamma + 1) * k_cism * self.data["e_51"] * 10 ** 51 / 4 / np.pi / rho_0 / r_cism ** 3) ** 0.5 / (
                10 ** 5) #V_s in White and Long

        def lin_velocity(t):
            """Linear velocity function to join ED phase to v_cism (reg_velocity)."""

            if n in (1,2,4):                                                                                          
                v_end = self.velocity_functions[n, "ED"](self.radius_functions[n, "ED"](self.calc["t_st"]))
            else:
                v_end = self.velocity_functions[n, "ED"](self.calc["t_st"])
            return (reg_velocity(1.1 * self.calc["t_st"]) - v_end) / 0.1 / self.calc["t_st"] * (t - self.calc["t_st"]) + v_end

        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]
    
##########################################################################################################################
    def _sedtay_solution(self, n, key):
        """Sedov-Taylor solution, which is the C/Tau=0 solution of the Cloudy ISM model 

        Args:
            n (int): n value for SNR
            key (str): determines which function is returned, see function_dict for possible values

        Returns:
            function: a function to calculate the parameter specified for SedovTaylor model
        """

        def radius(t):
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            e_0 = self.data["e_51"] * (10 ** 51)
            r = (((XI_0*e_0)/rho_0)**0.2)*(t**0.4)
            return r

        def velocity(t):
            rho_0 = self.data["n_0"] * M_H * self.data["mu_H"]
            e_0 = self.data["e_51"] * (10 ** 51)
            r = 0.4*(((XI_0*e_0)/rho_0)**0.2)*(t**-0.6)
            return r
                
        function_dict = {
            "r": radius,
            "v": velocity
        }
        return function_dict[key]


##########################################################################################################################
    def merger_time(self, n):
        """Gets merger times used to determine which phases occur for the s=0 case.

        Args:
            n (int): n value for SNR

        Returns:
            dict: dictionary of merger times for different phases of SNR evolution
        """

        t_mrg = {}
        if n == 0:
            t_mrg["ED"] = newton(lambda t: self.velocity_functions[n, "ED"](t) - self.calc["c_net"] * BETA, self.calc["t_st"])
        elif n in (1, 2, 4):                                                                     
            try:
                t_mrg["ED"] = newton(lambda t: self.velocity_functions[n, "ED"](
                    self.radius_functions[n, "ED"](t)) - self.calc["c_net"] * BETA, self.calc["t_st"])
            except ValueError:
                t_mrg["ED"] = np.inf
        else:
            t_mrg["ED"] = (self.calc["c_net"] * BETA / self.calc["v_ch"] * n / (n - 3) / (
                27 * self.cnst.l_ed ** (n - 2) / 4 / np.pi / n / (n - 3) / self.cnst.phi_ed * (10 * (n - 5) / 3 / (
                    n - 3)) ** ((n - 3) / 2)) ** (1 / n)) ** (-n / 3) * self.calc["t_ch"]
        if n == 0:
            t_mrg["ST"] = self.calc["t_ch"] / 1.42 * ((0.569 * self.calc["v_ch"] / BETA / self.calc["c_net"]) **
                                                      (5 / 3) + 0.254)
        else:
            t_mrg["ST"] = self.calc["t_ch"] * (((2 * XI_0 ** 0.5 * self.calc["v_ch"] / 5 / BETA / self.calc[
                "c_net"]) ** (5 / 3) - self.cnst.r_st ** 2.5) / XI_0 ** 0.5 + self.cnst.t_st)
        t_mrg["PDS"] = newton(lambda t: self.velocity_functions[n, "PDS"](t) - self.calc["c_net"] * BETA,
                              self.calc["t_pds"] * 2)
        try:
            t_mrg["MCS"] = newton(lambda t: self.velocity_functions[n, "MCS"](t) - self.calc["c_net"] * BETA,
                                  self.calc["t_mcs"])
        except RuntimeError:
            t_mrg["MCS"] = 0
        if self.data["model"] == "fel":
            phase = self.get_phase(self.data["t_fel"])
            r_0 = self.radius_functions[n, phase](self.data["t_fel"]) * PC_TO_KM
            v_0 = self.velocity_functions[n, phase](self.data["t_fel"])
            alpha_1 = (2 - self.calc["gamma_0"] + ((2 - self.calc["gamma_0"]) ** 2 + 4 * (self.data["gamma_1"] - 1)) **
                       0.5) / 4
            t_mrg["FEL"] = (((v_0 / BETA / self.calc["c_net"]) ** (-(4 - 3 * alpha_1) / 3 / (alpha_1 - 1)) - 1) *
                           r_0 / (v_0 * YR_TO_SEC) / (4 - 3 * alpha_1) + self.data["t_fel"])
        else:
            t_mrg["FEL"] = "N/A"
        t_mrg["HLD"] = self.data["t_hld"]
        if (self.data["model"] == "cism"):
            t_mrg["CISM"] = (K_DICT[self.data["c_tau"]] * self.data["e_51"] * 10 ** 51 / 4 / np.pi / self.data["mu_H"] /
                           M_H / self.data["n_0"]) ** (1 / 3) * (4 / 5 * (4 / 45) ** 0.2) ** (5 / 6) / \
                          (self.calc["c_net"] * 10**5 * 2) ** (5 / 3) / YR_TO_SEC
            if t_mrg["CISM"] < self.calc["t_st"] * 1.1:
                t_mrg["CISM"] = newton(lambda t: self.velocity_functions[n, "CISM"](t) - self.calc["c_net"] * BETA, self.calc["t_st"])
        else:
            t_mrg["CISM"] = "N/A"
        return t_mrg

#######################################################################################################################
    def get_phases(self):
        """Get phases that occur for the current input parameters and update the plot dropdown options accordingly.

        Returns:
            list: list of phases that occur for given input parameters
        """

        dropdown_values = ["Current", "Reverse Shock Lifetime", "ED-ST", "PDS", "MCS", "FEL", "HLD"]
        if self.data["s"] == 2:
            phases = "s2",
            dropdown_values = ["Current"]
        elif self.data["model"] == "fel":
            dropdown_values.remove("HLD")
            if self.data["t_fel"] == round(self.calc["t_st"]):
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED", "FEL")
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
            elif self.data["t_fel"] <= min(self.calc["t_pds"], self.calc["t_mrg"]["ST"]):
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "FEL")
            elif self.data["t_fel"] < min(self.calc["t_mcs"], self.calc["t_mrg"]["PDS"]):
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "PDS", "FEL")
            else:
                phases = ("ED", "ST", "PDS", "MCS", "FEL")
        elif self.data["model"] == "hld":
            dropdown_values.remove("FEL")
            dropdown_values.remove("PDS")
            dropdown_values.remove("MCS")
            if self.calc["t_c"] * 0.1 < self.calc["t_st"]:
                phases = ("ED", "HLD")
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
            else:
                phases = ("ED", "ST", "HLD")
        elif (self.data["model"] == "cism"):
            dropdown_values.remove("FEL")
            dropdown_values.remove("PDS")
            dropdown_values.remove("HLD")
            dropdown_values[dropdown_values.index("ED-ST")] = "ED-CISM"
            if self.calc["t_mcs"] < self.calc["t_mrg"]["CISM"]:
                phases = ("ED", "CISM", "MCS")
            else:
                dropdown_values.remove("MCS")
                phases = ("ED", "CISM")
        else:
            dropdown_values.remove("HLD")
            dropdown_values.remove("FEL")
            if self.calc["t_pds"] > self.calc["t_mrg"]["ST"]:
                dropdown_values.remove("PDS")
                dropdown_values.remove("MCS")
                phases = ("ED","ST")
            elif self.calc["t_mcs"] > self.calc["t_mrg"]["PDS"]:
                dropdown_values.remove("MCS")
                phases = ("ED", "ST", "PDS")
            else:
                phases = ("ED", "ST", "PDS", "MCS")
            if "ST" in phases and self.calc["t_st"] > self.calc["t_mrg"]["ED"]:
                dropdown_values[dropdown_values.index("ED-ST")] = "ED"
                phases = tuple([phase for phase in phases if phase != "ST"])
                self.widgets["model"].input["cism"].config(state="disabled")
        if self.data["s"] != 2 and str(self.widgets["model"].input["cism"].cget("state")) == "disabled" and self.calc["t_st"] < self.calc["t_mrg"]["ED"]:
            self.widgets["model"].input["cism"].config(state="normal")
        old = self.widgets["range"].get_value()
        self.widgets["range"].input.config(values=tuple(dropdown_values))
        
        # Set dropdown value to current if the old value is no longer an option
        if old not in dropdown_values and old != "Custom":
            self.widgets["range"].value_var.set("Current")
            self.data["range"] = "Current"
        return phases

##########################################################################################################################
    def _init_functions(self):
        """Create the function dictionaries used to calculate time, radius, and velocity. Note this function only needs
        to be called once.
        """

        self.radius_functions = {}
        self.velocity_functions = {}
        self.time_functions = {}

        for n in VALUE_DICT_S0:
            self.radius_functions[n, "PDS"] = self._pds_solution(n, "r")
            self.velocity_functions[n, "PDS"] = self._pds_solution(n, "v")
            self.radius_functions[n, "MCS"] = self._mcs_solution(n, "r")
            self.velocity_functions[n, "MCS"] = self._mcs_solution(n, "v")
            self.radius_functions[n, "FEL"] = self._fel_solution(n, "r")
            self.velocity_functions[n, "FEL"] = self._fel_solution(n, "v")
            self.radius_functions[n, "HLD"] = self._hld_solution(n, "r")
            self.velocity_functions[n, "HLD"] = self._hld_solution(n, "v")
            self.radius_functions[n, "CISM"] = self._cism_solution(n, "r")
            self.velocity_functions[n, "CISM"] = self._cism_solution(n, "v")
            if n == 0:
                self.radius_functions[n, "sedTay"] = self._sedtay_solution(n, "r")
                self.velocity_functions[n, "sedTay"] = self._sedtay_solution(n, "v")
                
            if n != 0:
                self.radius_functions[n, "ST"] = self._st_solution("r")
                self.radius_functions[n, "late"] = self._st_solution("rr")
                self.velocity_functions[n, "ST"] = self._st_solution("v")
                self.velocity_functions[n, "late"] = self._st_solution("vr")
                if n not in (1, 2, 4):                                                                     
                    self.radius_functions[n, "ED"] = self._cn_solution(n, "r")
                    self.radius_functions[n, "early"] = self._cn_solution(n, "rr")
                    self.velocity_functions[n, "ED"] = self._cn_solution(n, "v")
                    self.velocity_functions[n, "early"] = self._cn_solution(n, "vr")
                else:
                    self.time_functions[n, "ED"] = self._ed_solution(n, "t")
                    self.time_functions[n, "early"] = self._ed_solution(n, "tr")
                    self.radius_functions[n, "ED"] = lambda t, n=n: brentq(
                        lambda r, t: self.time_functions[n, "ED"](r) - t, 0, self.cnst.r_st * self.calc["r_ch"], t)
                    self.radius_functions[n, "early"] = lambda t, n=n: brentq(
                        lambda r, t: self.time_functions[n, "early"](r) - t, 0, self.cnst.r_rchg * self.calc["r_ch"], t)
                    self.velocity_functions[n, "ED"] = self._ed_solution(n, "v")
                    self.velocity_functions[n, "early"] = self._ed_solution(n, "vr")
            else:
                self.radius_functions[n, "ED"] = self._s0n0_solution("r", "ED")
                self.radius_functions[n, "ST"] = self._s0n0_solution("r", "ST")
                self.radius_functions[n, "early"] = self._s0n0_solution("rr", "early")
                self.radius_functions[n, "late"] = self._s0n0_solution("rr", "late")
                self.velocity_functions[n, "ED"] = self._s0n0_solution("v", "ED")
                self.velocity_functions[n, "ST"] = self._s0n0_solution("v", "ST")
                self.velocity_functions[n, "early"] = self._s0n0_solution("vr", "early")
                self.velocity_functions[n, "late"] = self._s0n0_solution("vr", "late")

        for n in (0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14):                                                                      
            if n >= 6:
                self.radius_functions[n, "s2"] = self._s2ngt5_solution(n, "r")
                self.radius_functions[n, "s2r"] = self._s2ngt5_solution(n, "rr")
                self.velocity_functions[n, "s2"] = self._s2ngt5_solution(n, "v")
                self.velocity_functions[n, "s2r"] = self._s2ngt5_solution(n, "vr")
            else:
                self.time_functions[n, "s2"] = self._s2nlt3_solution(n, "t")
                self.time_functions[n, "s2r"] = self._s2nlt3_solution(n, "tr")
                self.radius_functions[n, "s2"] = lambda t, n=n: brentq(
                    lambda r, t: self.time_functions[n, "s2"](r) - t, 0, 1 / 1.5 ** 2 / (
                        3 - self.data["n"]) * self.calc["r_ch"] - 1, t)
                self.radius_functions[n, "s2r"] = lambda t, n=n: brentq(
                    lambda r, t: self.time_functions[n, "s2r"](r) - t, 0, 1 / 1.5 ** 2 / (
                        3 - self.data["n"]) / 1.19 * self.calc["r_ch"] - 1, t)
                self.velocity_functions[n, "s2"] = self._s2nlt3_solution(n, "v")
                self.velocity_functions[n, "s2r"] = self._s2nlt3_solution(n, "vr")

#####################################################################################################################
#####################################################################################################################
class SNREmissivity:
    """Calculate and store emissivity data for a specific instance of a SuperNovaRemnant.

    Attributes:
        data (dict): dictionary of input parameters for emissivity model
        nrg_to_edt (float): conversion factor to get unitless energy divided by shock temperature from energy
        plots (dict): dictionary of OutputPlot instances shown on emissivity window
        cnst (dict): constants used in emissivity calculations
        root (str): ID of emissivity window, used to access widgets
    """
######################################################################################################################
    def __init__(self, snr, root):
        """Create instance of SNREmissivity class. Note that caching is used for functions when possible to speed up
        calculations. In some cases, scalar versions of vector functions are created so that a cache can be used (see
        scalar_temperature and scalar_density).

        Args:
            snr (SuperNovaRemnant): SuperNovaRemnant instance, used to pass input parameters to emissivity model
            root (str): ID of emissivity window, used to access widgets
        """

        keys = ("abundance", "ej_abundance", "mu_H", "mu_e", "mu_I", "mu_H_ej", "mu_e_ej", "mu_I_ej", "Z_sq", "Z_sq_ej",
                "n_0")
        self.data = {key: snr.data[key] for key in keys}
        # r_c is the radius of the contact discontinuity, em_point lists points of difficulty in EM integral
        self.data.update({"r_c": 0, "em_point": None, "radius": snr.calc["r"] * PC_TO_KM * 100000,
                          "T_s": snr.calc["T"], "r_min": 0})
        self.nrg_to_edt = KEV_TO_ERG / BOLTZMANN / self.data["T_s"]
        # Graphs defined in snr.py
        self.plots = {}
        self.cnst = {"intensity": (2 * (4 * self.data["n_0"]) ** 2 * self.data["mu_H"] ** 2 / self.data["mu_e"] /
                                   self.data["mu_I"] * self.data["radius"] / self.data["T_s"] ** 0.5)}
        self.cnst["spectrum"] = 8 * np.pi ** 2 * self.data["radius"] ** 2 * self.cnst["intensity"]
        self.cnst["luminosity"] = self.data["T_s"] * BOLTZMANN / PLANCK * self.cnst["spectrum"]
        self.root = root
        
        if(((snr.data["t"] <= snr.calc["t_st"]) or (snr.data["s"] == 2)) and snr.data["model"] != "sedtay"):
            self.data["model"] = "chev"
            if snr.data["s"] == 0:
                if snr.data["n"] >= 6:                             
                    self.data["r_c"] = 1 / S0_CHEV_RMAX[snr.data["n"]]    #where 1.181 is is the max value for r on each file
                    self.data["r_min"] = S0_CHEV_RMIN[snr.data["n"]] * self.data["r_c"] #where 0.935 is is the min value for r on each file
                    # r_difficult gives radius values at which pressure/temperature change dramatically
                    self.data["r_difficult"] = [self.data["r_c"]]
                    self.data["em_point"] = [self.data["r_difficult"], None]
            else:  #s=2 case
                self.data["r_c"] = 1 / S2_CHEV_RMAX[snr.data["n"]]
                self.data["r_min"] = S2_CHEV_RMIN[snr.data["n"]] * self.data["r_c"]
                self.data["r_difficult"] = [0.9975 * self.data["r_c"], 1.02 * self.data["r_c"]]
                self.data["em_point"] = [[self.data["r_difficult"][0]], [self.data["r_difficult"][1]]]
            # ej_correction used to account for different composition of reverse shock (ejecta vs. ISM abundances)
            self.data["ej_correction"] = (self.data["Z_sq_ej"] / self.data["Z_sq"] * self.data["mu_I"] /
                                          self.data["mu_I_ej"] * self.data["mu_e"] / self.data["mu_e_ej"])
            # Get radius, density, and pressure profiles
            lines = np.loadtxt("data/Chev_s{0:.0f}n{1}.txt".format(snr.data["s"], snr.data["n"]))
            radius = lines[:, 0] * self.data["r_c"]
            density = lines[:, 1]
            pressure = lines[:, 2]
            mu_norm = np.ones_like(radius)
            mu_norm[np.where(radius < self.data["r_c"])] = (1 / self.data["mu_e"] + 1 / self.data["mu_I"]) / (
                1 / self.data["mu_e_ej"] + 1 / self.data["mu_I_ej"])
            temperature = pressure / density * mu_norm
            # Interpolate density and temperature profiles to create density and temperature functions
            self.vector_density = lambda x: np.interp(x, radius, density)
            self.vector_temperature = lambda x: np.interp(x, radius, temperature)
            self.scalar_temperature = functools.lru_cache(maxsize=None)(self.vector_temperature)
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
            # Overwrite default class methods
            self._s_lim = self._chev_s_lim
            self._s_point = self._chev_s_point
            self._opt_dict = self._chev_opt_dict
            
        elif (snr.data["model"] == "cism" or snr.data["model"] == "sedtay" ):
            self.data["model"] = "cism"
            self.data["c_tau"] = snr.data["c_tau"]
            self.data["em_point"] = []
            # Get temperature and density profiles
            self.vector_temperature = self._file_interpTemp("WLsoln")
            self.vector_density = self._file_interpDens("WLsoln")
            self.scalar_temperature = functools.lru_cache(maxsize=None)(self.vector_temperature)
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
        elif ((snr.data["model"] == "cism") and (snr.data["c_tau"] == -1)):
            self.data["model"] = "cism"
            self.data["c_tau"] = snr.data["c_tau"]
            self.data["em_point"] = []
            self.vector_temperature = self._file_interpTemp("WLsoln", self.data["T_s"])
            self.scalar_temperature = self._sedov_scalar_temp
            self.vector_density = self._file_interpDens("WLsoln")
            self.scalar_density = functools.lru_cache(maxsize=None)(self.vector_density)
        else:
            self.data["model"] = "sedov"
            self.vector_temperature = self._sedov_vector_temp
            self.scalar_temperature = self._sedov_scalar_temp
            self.vector_density = self._sedov_vector_density
            self.scalar_density = self._sedov_scalar_density
        # Cache instance methods
        self._jnu_scaled = functools.lru_cache(maxsize=None)(self._jnu_scaled)
        self._intensity_integrand = functools.lru_cache(maxsize=None)(self._intensity_integrand)
        self._luminosity_integrand = functools.lru_cache(maxsize=None)(self._luminosity_integrand)

########################################################################################################################
    def _file_interpTemp(self, prefix, multiplier=1):
        """Read CISM data file and linearly interpolate temperature and density data.

        Args:
            prefix (str): file name excluding the C/tau value and extension
            multiplier (float): constant multiplied by all y values found

        Returns:
            function: interpolating function for data in specified file
        """

        lines = np.genfromtxt("data/{}{}_xfgh.txt".format(prefix, self.data["c_tau"]), delimiter=" ")
        lines[:, 1] = multiplier * lines[:, 1] / lines[:, 2]
        return lambda x: np.interp(x, lines[:, 0], lines[:, 1])
    
########################################################################################################################
    def _file_interpDens(self, prefix, multiplier=1):
        """Read CISM data file and linearly interpolate temperature and density data.

        Args:
            prefix (str): file name excluding the C/tau value and extension
            multiplier (float): constant multiplied by all y values found

        Returns:
            function: interpolating function for data in specified file
        """

        lines = np.genfromtxt("data/{}{}_xfgh.txt".format(prefix, self.data["c_tau"]), delimiter=" ")
        lines[:, 2] = multiplier * lines[:, 2]
        return lambda x: np.interp(x, lines[:, 0], lines[:, 2])
    
########################################################################################################################
    def update_output(self):
        """Update all plots and output values."""

        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Lnu", (self.data["emin"], self.data["emax"]))
        self.update_plot("Inu", (0, 1))
        self.update_plot("temp", (self.data["r_min"], 1))
        self.update_plot("density", (self.data["r_min"], 1))
        em = self.emission_measure()
        lum = self.total_luminosity()
        if self.data["model"] == "chev":
            # Show forward and reverse shock values separately in addition to total values
            output = {"lum": lum[0], "lum_f": lum[1], "lum_r": lum[2], "em": em[0], "em_f": em[1], "em_r": em[2],
                      "Tem": em[3], "Tem_f": em[4], "Tem_r": em[5]}
        else:
            output = {"lum": lum, "em": em[0], "Tem": em[1]}
            
        gui.OutputValue.update(output, self.root, 0)
        

#######################################################################################################################
    def update_specific_intensity(self):
        """Update specific intensity plot."""

        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Inu", (0, 1))

#######################################################################################################################
    def update_luminosity_spectrum(self):
        """Update luminosity plot and total luminosity value(s)."""
        self.data.update(gui.InputParam.get_values(self.root))
        self.update_plot("Lnu", (self.data["emin"], self.data["emax"]))
        lum = self.total_luminosity()
        if self.data["model"] == "chev":
            # Show forward and reverse shock values separately
            output = {"lum": lum[0], "lum_f": lum[1], "lum_r": lum[2]}
        else:
            output = {"lum": lum}
        gui.OutputValue.update(output, self.root, 0)

#######################################################################################################################
    def update_plot(self, key, limits):
        """Get data and redraw plot within given x-axis limits.

        Args:
            key (str): key used to identify which plot to update (Inu, Lnu, temp, or density)
            limits (tuple): x-axis limits for plot
        """

        plot = self.plots[key]
        plot.clear_plot()
        x_data = np.linspace(*limits, 150)
        # Get new data using function associated with plot
        y_data = plot.properties["function"](x_data)
        plot.add_data(x_data, y_data, color=plot.properties["color"])
        plot.display_plot(limits=limits)

#######################################################################################################################
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _sedov_scalar_temp(cls, x):
        """Temperature profile of Sedov phase SNR.

        Args:
            x (float): normalized radius of SNR (r/r_total)

        Returns:
            float: normalized temperature at specified radius (T/T_shock)
        """

        if x < 0.4:
            return 0.4 ** -4.32
        else:
            return x ** -4.32

#######################################################################################################################
    @classmethod
    def _sedov_vector_temp(cls, x):
        """Temperature profile of Sedov phase SNR as a vector function.

        Args:
            x (np.ndarray): normalized radii of SNR (r/r_total)

        Returns:
            np.ndarray: normalized temperatures at specified radii (T/T_shock)
        """

        lower_length = x[np.where(x < 0.4)].size
        upper = x[np.where(x >= 0.4)]
        return np.concatenate([np.full(lower_length, 0.4 ** -4.32), upper ** -4.32])

#######################################################################################################################
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _sedov_scalar_density(cls, x):
        """Density profile of Sedov phase SNR.

        Args:
            x (float): normalized radius of SNR (r/r_total)

        Returns:
            float: normalized density at specified radius (rho/rho_shock)
        """

        if x < 0.5:
            return 0.31 / cls._sedov_scalar_temp(x)
        else:
            return (0.31 + 2.774 * (x - 0.5) ** 3 + 94.2548 * (x - 0.5) ** 8.1748) / cls._sedov_scalar_temp(x)

#######################################################################################################################
    @classmethod
    def _sedov_vector_density(cls, x):
        """Density profile of Sedov phase SNR as a vector function.

        Args:
            x (float): normalized radii of SNR (r/r_total)

        Returns:
            float: normalized densities at specified radii (rho/rho_shock)
        """

        lower_length = x[np.where(x < 0.5)].size
        upper = x[np.where(x >= 0.5)]
        return np.concatenate([np.full(lower_length, 0.31), 0.31 + 2.774 * (upper - 0.5) ** 3 + 94.2548 * (
            upper - 0.5) ** 8.1748]) / cls._sedov_vector_temp(x)

#######################################################################################################################
    def _chev_s_lim(self, b, *args):
        """Get limits of integral over s (where s = (r^2 - b^2)^(1/2)) for ED phase emissivity model. Note *args is used
        since function is called by nquad, which also provides edt as a parameter.

        Args:
            b (float): normalized impact parameter

        Returns:
            tuple: lower and upper limits for integral over s
        """

        return 0 if b >= self.data["r_min"] else (self.data["r_min"] ** 2 - b ** 2) ** 0.5, (1 - b ** 2) ** 0.5

#########################################################################################################################
    @staticmethod
    def _s_lim(b, *args):
        """Get limits of integral over s (where s = (r^2 - b^2)^(1/2)). Note *args is used since function is called
        by nquad, which also provides edt as a parameter.

        Args:
            b (float): normalized impact parameter

        Returns:
            tuple: lower and upper limits for integral over s
        """

        return (0, (1 - b ** 2) ** 0.5)

##########################################################################################################################
    def _chev_s_point(self, b):
        """Get points of difficulty in integral over s.

        Args:
            b (float): normalized impact parameter

        Returns:
            list: points that could cause difficulty in the integral over s OR None if no such points exist
        """

        points = []
        for radius in self.data["r_difficult"]:
            if b < radius:
                points.append((radius ** 2 - b ** 2) ** 0.5)
        if b < self.data["r_c"]:
            points.append((self.data["r_c"] ** 2 - b ** 2) ** 0.5)
        if len(points) == 0:
            return None
        else:
            return points

##########################################################################################################################
    @staticmethod
    def _s_point(b):
        """Get points of difficulty in integral over s. This is the default method used as a placeholder until a more
        specific method is introduced for a given model, so no points are returned.

            Args:
                b (float): normalized impact parameter

            Returns:
                list: points that could cause difficulty in the integral over s OR None if no such points exist
        """

        return None

###########################################################################################################################
    def emission_measure(self):
        """Get emission measure and emission weighted temperature for the current emissivity model. (Commented lines can
        print off values used in density_finder.py)

        Returns:
            tuple: emission measure, emission weighted temperature (note that three of each value are given for the ED
                   phase model in the order total, forward shock, and reverse shock)
        """

        integrand = lambda x: self.scalar_density(x) ** 2 * 4 * np.pi * x ** 2
        temp_integrand = lambda x: integrand(x) * self.scalar_temperature(x)
        if self.data["model"] == "chev":
            em_dimless = (quad(integrand, self.data["r_min"], self.data["r_c"], epsabs=1e-5,
                          points=self.data["em_point"][0])[0],
                          quad(integrand, self.data["r_c"], 1, epsabs=1e-5, points=self.data["em_point"][1])[0])
            em_f = (16 * self.data["n_0"] ** 2 * self.data["radius"] ** 3 * self.data["mu_H"] * em_dimless[1] /
                    self.data["mu_e"])
            em_r = (16 * self.data["n_0"] ** 2 * self.data["radius"] ** 3 * self.data["mu_H"] ** 2 * em_dimless[0] /
                    self.data["mu_H_ej"] / self.data["mu_e_ej"])
            em_temp = (quad(temp_integrand, self.data["r_min"], self.data["r_c"], epsabs=1e-5,
                            points=self.data["em_point"][0])[0] * self.data["T_s"],
                       quad(temp_integrand, self.data["r_c"], 1, epsabs=1e-5,
                            points=self.data["em_point"][1])[0] * self.data["T_s"])
            em_temp_tot = (em_temp[0] + em_temp[1]) / (em_dimless[0] + em_dimless[1])
            return em_f + em_r, em_f, em_r, em_temp_tot, em_temp[1] / em_dimless[1], em_temp[0] / em_dimless[0]
        else:
            em_dimless = quad(integrand, self.data["r_min"], 1, epsabs=1e-5, points=self.data["em_point"])[0]
            return (em_dimless * 16 * self.data["n_0"] ** 2 * self.data["mu_H"] / self.data["mu_e"] *
                    self.data["radius"] ** 3,
                    quad(temp_integrand, self.data["r_min"], 1, epsabs=1e-5, points=self.data["em_point"])[0] /
                    em_dimless * self.data["T_s"])

############################################################################################################################
    def _jnu_scaled(self, x, edt):
        """Get emission coefficient for thermal bremsstrahlung.

        Args:
            x (float): normalized radius (r/r_total)
            edt (float): energy (h \nu) divided by shock temperature (unitless)
            temp is in units of shock temperature

        Returns:
            float: emission coefficient at given x and edt
        """

        temp = self.scalar_temperature(x)
        val = ((np.log10(edt / temp) + 1.5) / 2.5)
        gaunt = (3.158 - 2.524 * val + 0.4049 * val ** 2 + 0.6135 * val ** 3 + 0.6289 * val ** 4 + 0.3294 * val ** 5 -
                 0.1715 * val ** 6 - 0.3687 * val ** 7 - 0.07592 * val ** 8 + 0.1602 * val ** 9 + 0.08377 * val ** 10)
        return 5.4e-39 * self.data["Z_sq"] / temp ** 0.5 * gaunt * np.exp(-edt / temp)

############################################################################################################################
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def norm_radius(b, s):
        """Get normalized radius from impact parameter and s value.

        Args:
            b (float): normalized impact parameter
            s (float): defined as (r^2-b^2)^(1/2)

        Returns:
            float: normalized radius value
        """
        return (s ** 2 + b ** 2) ** 0.5

############################################################################################################################
    def specific_intensity(self, b):
        """Get specific intensity at a given impact parameter and energy.

        Args:
            b (float): normalized impact parameter

        Returns:
            float: specific intensity at impact parameter b (and energy self.data["energy"])
        """
        edt = self.data["energy"] * self.nrg_to_edt
        integral = np.fromiter((quad(self._intensity_integrand, *self._s_lim(b_val), args=(b_val, edt),
                                     points=self._s_point(b_val))[0] for b_val in b), np.float)
        return self.cnst["intensity"] * integral

############################################################################################################################
    def _intensity_integrand(self, s, b, edt):
        """Integrand used in specific intensity integral.

        Args:
            s (float): defined as (r^2-b^2)^(1/2)
            b (float): normalized impact parameter
            edt (float): energy divided by shock temperature (unitless)

        Returns:
            float: value of integrand for given parameters
        """

        radius = self.norm_radius(b, s)
        # Multiplier needed to account for different composition of ejecta (for reverse shock)
        multiplier = self.data["ej_correction"] if radius < self.data["r_c"] else 1
        return self._jnu_scaled(radius, edt) * self.scalar_density(radius) ** 2 * multiplier

############################################################################################################################
    def _chev_opt_dict(self, b, *args):
        """Get option dictionary for luminosity integral over s in ED phase model.

        Args:
            b (float): normalized impact parameter
            *args: unused, provided since nquad provides edt as an additional parameter

        Returns:
            dict: options for luminosity integral over s
        """

        points = self._chev_s_point(b)
        if points is None:
            return {}
        else:
            return {"points": points}

#############################################################################################################################
    @staticmethod
    def _opt_dict(b, *args):
        """Get option dictionary for luminosity integral over s. Used as a placeholder until a new function is defined
        for a specific case.

        Args:
            b (float): normalized impact parameter
            *args: unused, provided since nquad provides edt as an additional parameter

        Returns:
            dict: options for luminosity integral over s (empty since function is a placeholder)
        """

        return {}

#############################################################################################################################
    def luminosity_spectrum(self, energy):
        """Get luminosity at a given energy for luminosity spectrum.

        Args:
            energy (float): photon energy in keV

        Returns:
            float: luminosity at given energy
        """

        edt_array = energy * self.nrg_to_edt
        integral = np.fromiter((nquad(self._luminosity_integrand, [self._s_lim, [0, 1]],
                                      args=(edt,), opts=[self._opt_dict, {}])[0] for edt in edt_array), np.float)
        return integral * self.cnst["spectrum"]

#############################################################################################################################
    def total_luminosity(self):
        """Get total luminosity over a given energy range.

        Returns:
            tuple: total luminosity, luminosity of forward shock, luminosity of reverse shock (for ED phase model)
            float: total luminosity (for other models)
        """

        edt_min = self.data["emin"] * self.nrg_to_edt
        edt_max = self.data["emax"] * self.nrg_to_edt
        if self.data["model"] == "chev":
            lum_r = nquad(lambda s, b, edt: self._luminosity_integrand(s, b, edt) * self._reverse(s, b),
                          [self._s_lim, [0, 1], [edt_min, edt_max]],
                          opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0] * self.cnst["luminosity"]
            lum_f = nquad(lambda s, b, edt: self._luminosity_integrand(s, b, edt) * self._forward(s, b),
                          [self._s_lim, [0, 1], [edt_min, edt_max]],
                          opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0] * self.cnst["luminosity"]
            return lum_r + lum_f, lum_f, lum_r
        else:
            integral = nquad(self._luminosity_integrand, [self._s_lim, [0, 1], [edt_min, edt_max]],
                             opts=[self._opt_dict,{},{"points": [0.001, 0.1, 1, 10, 100]}])[0]
            return integral * self.cnst["luminosity"]

##############################################################################################################################
    def _luminosity_integrand(self, s, b, edt):
        """Integrand of luminosity integral in total_luminosity and luminosity_spectrum.

        Args:
            s (float): defined as (r^2-b^2)^(1/2)
            b (float): normalized impact parameter
            edt (float): energy divided by shock temperature (unitless)

        Returns:
            float: integrand value for given parameters
        """

        radius = self.norm_radius(b, s)
        multiplier = self.data["ej_correction"] if radius < self.data["r_c"] else 1
        return b * self._jnu_scaled(radius, edt) * self.scalar_density(radius) ** 2 * multiplier

#############################################################################################################################
    def _forward(self, s, b):
        """Isolate forward shock values.

        Args:
            s (float): defined as (r^2-b^2)^2
            b (float): normalized impact parameter

        Returns:
             int: 1 if position is part of forward shock, 0 otherwise
        """

        return 1 if self.norm_radius(b, s) > self.data["r_c"] else 0

############################################################################################################################
    def _reverse(self, s, b):
        """Isolate reverse shock values.

        Args:
            s (float): defined as (r^2-b^2)^2
            b (float): normalized impact parameter

        Returns:
             int: 1 if position is part of reverse shock, 0 otherwise
        """

        return 1 if self.norm_radius(b, s) <= self.data["r_c"] else 0

##############################################################################################################################