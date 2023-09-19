"""SNRPY File IO Calculation Manager

Handles the IO of applying the SNRPY formulae to an input file and returns the outputs as a file

Authors: Bryson Lawton
Version: Dec 2019
"""
#import snr_plot as plt
#import math
#from tkinter import Tk
import csv
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

modelNames = ['stdfwdshock','stdrevshock','cloudy','sedov']
inputFileDatastruct = []
successfulFileParse = False

###########################################################################################################################
def openFileBrowser(window, SNR_INV):
    global successfulFileParse
    filepath = askopenfilename(parent = window.root, title = "Select a CSV file", filetypes = (("CSV Files","*.csv"),("All Files","*.*"))) # show an "Open" dialog box and return the path to the selected file
    if filepath != "":
        successfulFileParse = False
        if not filepath.endswith('.csv'):
            messagebox.showinfo("Invalid File Selected", "Filetype must be a CSV file.", parent = window.root)
        else:
            parseInverseDatafile(window, filepath)
            if (successfulFileParse):
                calcAndWriteToOutputFile(window, SNR_INV, filepath)
                


###########################################################################################################################     
def parseInverseDatafile(window, filepath):
    global successfulFileParse, inputFileDatastruct
    with open(filepath) as inputFile:
        inputFile_reader = csv.reader(inputFile, delimiter = ',')
        errorMessage = ""
        inputFileDatastruct = []
        rowData = []
        #Check if datafile in proper format before running through calculations
        lineCount = 0
        for row in inputFile_reader:
            rowData = []
            if (len(row) >= 5):
                modeltype = row[1].replace(" ","").lower()
                
                #For the standard forward shock case-----------------------------------------------------------------------------------------
                if (modeltype == "stdfwdshock"):
                    if (len(row) == 8):
                        for x in range(8):
                            entry = row[x].strip()
                            #Check user defined label
                            if (x == 0):
                                if len(entry) > 100:
                                    errorMessage = "User label on line " + str(lineCount+1) + " must be less than 100 characters."
                                    break
                            #Set the model name    
                            elif (x == 1):
                                entry = "standard_forward"
                            #Check the s and n values    
                            elif (x <= 3):
                                try:
                                    entryNum = int(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not an integer."
                                    break
                                if (x == 2): 
                                    if (entryNum != 0 and entryNum != 2):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be either a 0 or 2."
                                        break
                                else:
                                    if (entryNum < 6 or entryNum > 14):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be between 6 and 14 (inclusive)."
                                        break
                            #Check the floating point values        
                            else:
                                try:
                                    entryNum = float(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not a number."
                                    break
                                if (entryNum <= 0):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be greater than 0."
                                        break
                                    
                            rowData.append(entry)
                        if errorMessage != "":
                            break
                    else:
                        errorMessage = "Eight data values necessary on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                        break
                #For the standard reverse shock case-----------------------------------------------------------------------------------------                        
                elif (modeltype == "stdrevshock"):
                    if (len(row) == 8):
                        for x in range(8):
                            entry = row[x].strip()
                            #Check user defined label
                            if (x == 0):
                                if len(entry) > 100:
                                    errorMessage = "User label on line " + str(lineCount+1) + " must be less than 100 characters."
                                    break
                            #Set the model name    
                            elif (x == 1):
                                entry = "standard_reverse"
                            #Check the s and n values    
                            elif (x <= 3):
                                try:
                                    entryNum = int(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not an integer."
                                    break
                                if (x == 2): 
                                    if (entryNum != 0 and entryNum != 2):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be either a 0 or 2."
                                        break
                                else:
                                    if (entryNum < 6 or entryNum > 14):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be between 6 and 14 (inclusive)."
                                        break
                            #Check the floating point values        
                            else:
                                try:
                                    entryNum = float(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not a number."
                                    break
                                if (entryNum <= 0):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be greater than 0."
                                        break
                                    
                            rowData.append(entry)
                        if errorMessage != "":
                            break
                    else:
                        errorMessage = "Eight data values necessary on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                        break
                    
                #For the cloudy case---------------------------------------------------------------------------------------------------------                    
                elif (modeltype == "cloudy"):
                    if (len(row) == 6):
                        for x in range(6):
                            entry = row[x].strip()
                            #Check user defined label
                            if (x == 0):
                                if len(entry) > 100:
                                    errorMessage = "User label on line " + str(lineCount+1) + " must be less than 100 characters."
                                    break
                            #Set the model name    
                            elif (x == 1):
                                entry = "cloudy_forward"
                            #Check the ctau value    
                            elif (x == 2):
                                try:
                                    entryNum = int(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not an integer."
                                    break
                                if (entryNum != 0 and entryNum != 1 and entryNum != 2 and entryNum != 4):
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be either a 0, 1, 2 or 4."
                                    break
                            #Check the floating point values        
                            else:
                                try:
                                    entryNum = float(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not a number."
                                    break
                                if (entryNum <= 0):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be greater than 0."
                                        break
                                    
                            rowData.append(entry)
                        if errorMessage != "":
                            break
                    else:
                        errorMessage = "Six data values necessary on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                        break
                    
                #For the sedov case-----------------------------------------------------------------------------------------------------------                         
                elif (modeltype == "sedov"):
                    if (len(row) == 5):
                        for x in range(5):
                            entry = row[x].strip()
                            #Check user defined label
                            if (x == 0):
                                if len(entry) > 100:
                                    errorMessage = "User label on line " + str(lineCount+1) + " must be less than 100 characters."
                                    break
                            #Set the model name    
                            elif (x == 1):
                                entry = "sedov"
                            #Check the floating point values        
                            else:
                                try:
                                    entryNum = float(entry)
                                except ValueError:
                                    errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " is not a number."
                                    break
                                if (entryNum <= 0):
                                        errorMessage = "Variable "+ str(x+1) + " on line " + str(lineCount+1) + " must be greater than 0."
                                        break
                                    
                            rowData.append(entry)
                        if errorMessage != "":
                            break
                    else:
                        errorMessage = "Five data values necessary on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                        break
                    
                #If the model name is not valid
                else:
                    errorMessage = "SNR model name invalid on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                    break
                #-------------------------------------------------------------------------------------------------------------------------------
            #If not enough variables in a line
            elif (len(row) < 5):
                if (len(row) != 0):
                    errorMessage = "There are not enough variables on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                    break
            #If too many variables in a line
            elif(len(row) > 8):
                errorMessage = "There are too many variables on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                break
            #Catch for other unforseen formatting errors
            else:
                errorMessage = "Formatting error on line " + str(lineCount+1) + ". \nPlease see formatting guidelines file."
                break
            inputFileDatastruct.append(rowData)
            lineCount += 1
        #If no errors found, finish parsing datafile
        if (errorMessage == ""):
            successfulFileParse = True
        #If errors found, clear datastruct and show error message to user
        else:
            inputFileDatastruct = []
            messagebox.showinfo("Invalid File Format", errorMessage, parent = window.root)
            
        inputFile.close()
        
###########################################################################################################################    
def calcAndWriteToOutputFile(window, SNR_INV, filepath):
    global inputFileDatastruct
    outputFilepath = filepath[:-4] + "_SNRPYoutput.csv"
    outputFile = open(outputFilepath, "w")
    outputFile.write("User Defined Label,Model Type,Age(yrs),Energy(ergs),ISM # Density(cm^-3),FS Radius(pc),FS Temp.(keV),FS Emission Measure(cm^-3),RS Radius(pc),RS Temp.(keV),RS Emission Measure(cm^-3),RS Reaches Core(yrs),RS Reaches Center(yrs)\n")
    for dataline in inputFileDatastruct:
        if (len(dataline) == 0):
            outputFile.write("\n")
        else:
            if(dataline[1] == "standard_forward"):
                lineDict = {"model_inv": dataline[1], "n_inv": int(dataline[3]), "s_inv": int(dataline[2]), "m_eject_inv": float(dataline[4]), "R_f_inv": float(dataline[5]), "Te_f_inv": float(dataline[6]), "EM58_f_inv": float(dataline[7])}
            elif(dataline[1] == "standard_reverse"):
                lineDict = {"model_inv": dataline[1], "n_inv": int(dataline[3]), "s_inv": int(dataline[2]), "m_eject_inv": float(dataline[4]), "R_f_inv": float(dataline[5]), "Te_r_inv": float(dataline[6]), "EM58_r_inv": float(dataline[7])}
            elif(dataline[1] == "cloudy_forward"):
                lineDict = {"model_inv": dataline[1], "ctau_inv": int(dataline[2]), "R_f_inv": float(dataline[3]), "Te_f_inv": float(dataline[4]), "EM58_f_inv": float(dataline[5])}
            elif(dataline[1] == "sedov"):
                lineDict = {"model_inv": dataline[1], "R_f_inv": float(dataline[2]), "Te_f_inv": float(dataline[3]), "EM58_f_inv": float(dataline[4])}
            outputFile.write(dataline[0] + "," + SNR_INV.outputFile_createLine(lineDict) + "\n")
    outputFile.close()
    messagebox.showinfo("", "Finished Generating Output Datafile", parent = window.root)
        
        
############################################################################################################################
        
        
        
        
        
        
        
        
        