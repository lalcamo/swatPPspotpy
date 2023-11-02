'''
Created by: Lucas Alcamo
Created on: 31.08.2023
Last edited on: XX.XX.2023
'''
import numpy as np
import os
from datetime import datetime
    
################### Run SWAT+ model and return outflow ################### 
# 
# Input: - parameter_values - Calibration parameters (np.array/list)
#        - path_TxtInOut    - Path to TextInOut folder (str)
#        - name_exe         - Name to python executable (str)
#        - channel_id       - ID of channel (int, default=1) 
# 
# Output: -data             -  Discharge at outflow (list)
##########################################################################

def run_swatP(path_TxtInOut, name_exe, parameter_values, channel_id=1):
    
    # print(parameter_values)

    # Give channel_id the default value if not defined or wrongly defined
    try:
        if channel_id == None or channel_id == 0 or type(channel_id) != int:
            channel_id = 1
    except NameError:
        channel_id = 1

    # Start time
    t_start = datetime.now()
    
    #######################################################
    # Edit calibration.cal file with given parameter_values 
    #######################################################
    with open(f"{path_TxtInOut}calibration_read.cal", "r") as file:
    
        lines = file.readlines()
        
        # Iterate through the lines and update parameter values 
        for i, line in enumerate(lines[3:]):
            
            # Two cases possible!- with and without HRU ranges 
            if len(line.strip().split())>=12:
                name_var, chg_typ, chg_val, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, obj_tot, HRU_lower, HRU_upper = line.strip().split()
                
                # Find respective spotpy name
                HRU_range = HRU_lower + '  ' + HRU_upper
                with open(f"{path_TxtInOut}HRU_link.txt", "r") as hru_link_file: 
                    lines_link = hru_link_file.readlines()
                    for line_link in lines_link[1:]:
                        if HRU_range and name_var in line_link:
                            name_var, HRU_lower, HRU_upper, name_spotpy = line_link.strip().split()
                
            else:
                name_var, chg_typ, chg_val, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, obj_tot = line.strip().split()
                HRU_range = ''
                
                # Find sespective spotpy name
                param_names = list(parameter_values.name)
                param_name  = [str(param_name) for param_name in param_names if name_var in param_name]
                name_spotpy = param_name[0]
            
            # Select respective parameter
            param_value = parameter_values[name_spotpy]
                
            chg_val = param_value
            # ROUNDING #### HAS INFLUNCE ON SAMPLING IS THIS OK? MAYBE CHANGE TO BE DEPENDENT ON SIZE OF VALUE
            chg_val = round(chg_val, 5)
            
            # Write new line with new parameter value (chg_val)
            line_new = f"{name_var:20}{chg_typ:>10}{chg_val:>14}{'0':>10}{'0':>11}{'0':>11}{'0':>10}{'0':>10}{'0':>10}{'0':>10}{obj_tot:>10}  {HRU_range}\n"
            
            # Replace the original line with the modified line (WHATCH OUT LINE+3!!!)
            lines[i+3] = line_new
            
    
    # Save calibration file
    with open(f"{path_TxtInOut}calibration.cal", "w") as file:
        file.writelines(lines)
    

    ###########
    # Run Model 
    ###########
    # Change to model directory
    os.chdir(path_TxtInOut)
    
    # Run SWAT+
    os.system(path_TxtInOut + name_exe)
    

    #############
    # Read output
    # Monthly output 
    #############
    data = [] 
    with open(f"{path_TxtInOut}channel_sd_mon.txt", "r") as file:
        for _ in range(3):
            next(file)
        for line in file:
            columns = line.split()  # Split the line into columns
            unit    = int(columns[4]) 
            if unit == channel_id:
                discharge = float(columns[47])  
                data.append(discharge)  

    #######
    # Timer
    #######

    # End time 
    t_end = datetime.now()
    
    # Print elapsed time
    print('Run completed, elapsed time: ' , t_end-t_start)

    return data