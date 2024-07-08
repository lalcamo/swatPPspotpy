'''
Created by: Lucas Alcamo
Created on: 31.08.2023
Last edited on: XX.XX.2023
'''
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from datetime import datetime
from mpi4py import MPI
import shutil

################### Write empty calibration.cal (and HRU_link.txt) file ################### 
# 
# Input: - 
#        - 
# Output: - None
###########################################################################################

def write_empty_cal_file(path_shapes, name_HRUshp, path_TxtInOut, df_params):
    
    # Read HRUs file to link subbasin numbers with LSUIDs
    HRU_file = gpd.read_file(path_shapes + name_HRUshp, ignore_geometry=True)

    # Write blank calibration.cal file
    header_line = 'calibration.cal: written by SWAT++SPOTPY\n'
    second_line = ' '
    column_line = 'cal_parm               chg_typ       chg_val     conds  soil_lyr1  soil_lyr2       yr1       yr2      day1      day2   obj_tot\n'
    lines       = [header_line,second_line,column_line]

    # For HRU link file
    header_link = 'HRU_link: written by SWAT++SPOTPY\n'
    lines_link  = [header_link]
    cal_params  = pd.read_csv(path_TxtInOut + 'cal_parms.cal', header=1, skiprows=[1], delim_whitespace=True, index_col=None)


    for ind in df_params.index:
        # Define parameters
        name_var       = df_params.at[ind,'name']
        name_spotpy    = df_params.at[ind,'name_spotpy']
        chg_typ        = df_params.at[ind,'change_type']
        subbasins_list = df_params.at[ind,'subbasins']
        obj_tot        = 0 
        HRU_range      = ''

        # If subbasins were defined
        if subbasins_list != None:

            for subbasin_nr in subbasins_list:

                # Range of HRUs within the subbasin
                # Check if parameter is a aquifer parameter
                if cal_params[cal_params.name == name_var].obj_typ.values[0] == 'aqu':
                    HRU_range = str(subbasin_nr) + '  -' + str(subbasin_nr)
    
                else:
                    HRUs_list = list(HRU_file[HRU_file['Subbasin'] == subbasin_nr]['HRUS'].values)
                    HRUs_list = [int(hru_nr) for hru_nr in HRUs_list]
                    HRUs_list.sort();
                    HRU_range = str(min(HRUs_list)) + '  -' + str(max(HRUs_list))

                # Object total is 2 when subbasin selected (range) 
                obj_tot   = 2

                # Add line for each subbasin and parameter, set values of fixed parameters, else 'replace'
                if pd.isna(df_params.at[ind,'upper_bound']) == True:
                    chg_val = df_params.at[ind,'lower_bound']
                else:
                    chg_val = 'replace'
                line = f"{name_var:20}{chg_typ:>10}{chg_val:>14}{'0':>10}{'0':>11}{'0':>11}{'0':>10}{'0':>10}{'0':>10}{'0':>10}{obj_tot:>10}  {HRU_range}\n"
                lines.append(line)

                # Write lines to link subbasins to HRU ranges
                line_link = f'{name_var} {HRU_range} {name_spotpy}\n'
                lines_link.append(line_link)

        else:
            # Add one line for each parameter if no subbains were defined
            if pd.isna(df_params.at[ind,'upper_bound']) == True:
                chg_val = df_params.at[ind,'lower_bound']
            else:
                chg_val = 'replace'
            line = f"{name_var:20}{chg_typ:>10}{chg_val:>14}{'0':>10}{'0':>11}{'0':>11}{'0':>10}{'0':>10}{'0':>10}{'0':>10}{obj_tot:>10}\n"
            lines.append(line);    

    # Nr of parameters to calibrate
    lines[1] = f"{str(len(lines)-3)}\n"

    # Write blank calibration file 
    with open(f"{path_TxtInOut}calibration_read.cal", "w") as file:
        file.writelines(lines);

    # Write file to link subbasins to HRU ranges
    with open(f"{path_TxtInOut}HRU_link.txt", "w") as file:
        file.writelines(lines_link);

    return None
    
####################### Prep parallel processing ####################### 
#
########################################################################

def prep_parallel_processing(mpi, path_Default, path_TxtInOut, name_TxtInOut):
    # Prep for parallel processing 
    if mpi == True:
        #OS must be windows
        comm = MPI.COMM_WORLD 
        core_nr = str(comm.Get_rank()+1) # +1 to prevent zero
        # Create a path and copy TxtInOut for each new core_nr seen
        path_core = path_Default + name_TxtInOut + '_core' + str(core_nr)
        # Copy TxtInOut folder if it doesn't exist!
        if os.path.exists(path_core + os.sep) == False:
            try:
                print('copy: ', path_Default + name_TxtInOut)
                print('to: ', path_core)
                shutil.copytree(path_Default + name_TxtInOut, path_core)
            except WindowsError as e:
                print ("ERROR: WINDOWSERROR = ",e)
            except :
                print ("ERROR: Some other error happened")   
        # Define from which path to run 
        path_run = path_core + '/'
        
    else: 
        # Define from which path to run 
        path_run = path_TxtInOut
        
    return path_run

################### Get simulation dates from model ################### 
# Create date param based on time.sim and warmup period from print.prt
# 
# Input: - path_TxtInOut 
#        - flag_ts
# 
# Output: - dates             - list of all dates (pandas.core.indexes.datetimes.DatetimeIndex)
########################################################################

def get_sim_dates(path_TxtInOut, flag_ts):
    with open(f"{path_TxtInOut}print.prt", "r") as file:
        lines = file.readlines()
        nyskip, day_start , yrc_start, day_end, yrc_end, interval = lines[2].strip().split()
    with open(f"{path_TxtInOut}time.sim", "r") as file:
        lines = file.readlines()
        jul_day_strt, yr_strt, jul_day_end, yr_end, step = lines[2].strip().split()
    if jul_day_strt == '0':
        jul_day_strt = '1'
    if jul_day_end == '0':
        jul_day_end = '365'
    yr_strt_true = str(int(yr_strt) + int(nyskip))
    dddyyyy_strt = f"{int(jul_day_strt):03d}-{yr_strt_true}"
    yr_strt      = pd.to_datetime(dddyyyy_strt, format='%j-%Y', dayfirst=True)
    dddyyyy_end  = f"{int(jul_day_end):03d}-{yr_end}"
    yr_end       = pd.to_datetime(dddyyyy_end, format='%j-%Y', dayfirst=True)
    if flag_ts == 'day':
        freq = 'D'
    elif flag_ts == 'month':
        freq = 'M'
    else:
        print('Issue with flag_ts, daily ts assumed!')
        freq = 'D'

    dates = pd.date_range(start=yr_strt, end=yr_end, freq=freq)

    return dates
    
################### Run SWAT+ model and return outflow ################### 
# 
# Input: - parameter_values - Calibration parameters (np.array/list)
#        - path_TxtInOut    - Path to TextInOut folder (str)
#        - name_exe         - Name to python executable (str)
#        - channel_id       - ID of channel (int, default=1) 
# 
# Output: -data             -  Discharge at outflow (list)
##########################################################################

def run_swatP(path_TxtInOut, name_exe, parameter_values):
    
    print('...........starting sim')

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
                # Ignore fixed parameters
                if chg_val == 'replace': 
                    with open(f"{path_TxtInOut}HRU_link.txt", "r") as hru_link_file: 
                        lines_link = hru_link_file.readlines()
                        for line_link in lines_link[1:]:
                            if HRU_range in line_link and name_var in line_link:
                                name_var, HRU_lower, HRU_upper, name_spotpy = line_link.strip().split()
                
            else:
                name_var, chg_typ, chg_val, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, place_holder, obj_tot = line.strip().split()
                HRU_range = ''
                # Ignore fixed parameters
                if chg_val == 'replace':
                    # Find respective spotpy name
                    param_names = list(parameter_values.name)
                    param_name  = [str(param_name) for param_name in param_names if name_var in param_name]
                    name_spotpy = param_name[0]

            
            # Select respective parameter
            if chg_val == 'replace':
                param_value = parameter_values[name_spotpy]
                chg_val     = param_value
                chg_val     = round(chg_val, 5) # rounding 
            else: 
                chg_val = chg_val
                
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
    os.system(name_exe) # os.system(path_TxtInOut + name_exe) 
    
    #######
    # Timer
    #######

    # End time 
    t_end = datetime.now()
    
    # Print elapsed time
    print('\t...Run Info: Run completed, elapsed time: ' , t_end-t_start)

    return None

################### Get simulation output ################### 
#       
#############################################################

def return_sim_data(df_variables, path_TxtInOut):
    
    data_calib = []
    data_cv    = {}
    for i_var in range(len(df_variables)):
        
        data_loc  = []
        
        var       = df_variables.at[i_var, 'variable']
        temp_res  = df_variables.at[i_var, 'temp_res']
        var_loc   = df_variables.at[i_var, 'location']
        calib     = df_variables.at[i_var, 'calib']
        obs_file  = df_variables.at[i_var, 'obs_file']
        
        if temp_res == 'daily':
            suff_var = '_day'
        elif temp_res == 'monthly':
            suff_var = '_mon'
        else:
            suff_var = 'ERROR'
            print('VARIABLE ERROR: Selected temp. resolution of variables:',temp_res,'(options: "daily" or "monthly"')
        
        if var == 'discharge':
            variable_file = 'channel_sd'+suff_var
            var_col = 47
            id_col  = 6 # id column is "name"
            lsu_area_frac = [1.0]
            
        elif var == 'eta':
            with open(f"{path_TxtInOut}LSU_link.txt", "r") as lsu_link_file: 
                lines_link    = lsu_link_file.readlines()
                lsu_ids       = [int(line_link.strip().split()[0]) for line_link in lines_link[2:] if int(line_link.strip().split()[1]) == var_loc[0]]
                lsu_area_frac = [float(line_link.strip().split()[4]) for line_link in lines_link[2:] if int(line_link.strip().split()[1]) == var_loc[0]]
            var_loc       = lsu_ids # Overwrite var_loc from subbasin id to lsu ids
            variable_file = 'lsunit_wb'+suff_var
            var_col       = 14
            id_col        = 6 # id column is "name"

        elif var == 'sm':
            with open(f"{path_TxtInOut}LSU_link.txt", "r") as lsu_link_file: 
                lines_link    = lsu_link_file.readlines()
                lsu_ids       = [int(line_link.strip().split()[0]) for line_link in lines_link[2:] if int(line_link.strip().split()[1]) == var_loc[0]]
                lsu_area_frac = [float(line_link.strip().split()[4]) for line_link in lines_link[2:] if int(line_link.strip().split()[1]) == var_loc[0]]
            var_loc       = lsu_ids # Overwrite var_loc from subbasin id to lsu ids
            variable_file = 'lsunit_wb'+suff_var
            var_col       = 22
            id_col        = 6 # id column is "name"

        else:
            variable_file = 'ERROR'
            print('VARIABLE ERROR: Selected variable does not exist:', var,'(options: "discharge" or "eta")')
        
        with open(f"{path_TxtInOut}{variable_file}.txt", "r") as file:
            for _ in range(3):
                next(file)
            jday_prev  = 0
            for line in file:
                columns = line.split()  # Split the line into columns
                id = columns[id_col][3:]
                id = int(id.lstrip('0'))
                jday = int(columns[0]) 
        
                if jday != jday_prev:
                    if jday_prev != 0:
                        data_loc.append(data_value)
                    data_value = 0.0
        
                if id in var_loc:
                    for loc in var_loc:
                        if loc == id:
                            index = var_loc.index(id)
                            data_value += float(columns[var_col])*lsu_area_frac[index]
                            
                jday_prev = jday
            # Add last value
            data_loc.append(data_value)
        
        if calib == True:
            data_calib.append(data_loc)
        
        data_cv[obs_file]=data_loc

    # If only one variable timeseries
    if len(data_calib) == 1:
        data_calib = data_calib[0]
        
    return data_calib, data_cv

#################### Get observed data #################### 
#       
###########################################################

def return_obs_data(df_variables, path_obs, date):
    
    trueObs = []
    origObs = []
    
    for i_var in range(len(df_variables)):
        file  = df_variables.at[i_var, 'obs_file']
        calib = df_variables.at[i_var, 'calib']
        if calib == True:
            flo_obs = pd.read_csv(path_obs + file, sep=';', index_col='Date', 
                                  parse_dates=True, date_format='%Y-%m-%d',na_values=[-9999.0])
            origObs.append(flo_obs)
            flo_obs = flo_obs.loc[date[0]:date[-1]]
            trueObs.append(list(flo_obs['Data']))
        
    # If only one observation
    if len(trueObs) == 1:
        trueObs = trueObs[0]

    return trueObs, origObs

def write_LSU_link_file(path_shapes, name_LSUshp, path_TxtInOut):
    # Read HRUs file and saves link file (for reading output data)
    LSU_file = gpd.read_file(path_shapes + name_LSUshp, ignore_geometry=True)
    LSU_file = LSU_file.sort_values('Subbasin')[['LSUID','Subbasin','Area','%Subbasin']]
    LSU_file['frac_Subbasin'] = LSU_file['%Subbasin']/100
    header_link = 'LSU_link: written by SWAT++SPOTPY\n'
    columns_link = 'LSUID Subbasin Area Area frac_Subbasin\n'
    lines_link  = [header_link, columns_link]
    lines_link += [f"{LSU_file.at[iloc, 'LSUID']} {LSU_file.at[iloc, 'Subbasin']} {LSU_file.at[iloc, 'Area']} {LSU_file.at[iloc, 'Area']} {LSU_file.at[iloc, 'frac_Subbasin']}\n"  
                   for iloc in range(len(LSU_file))]
    with open(f"{path_TxtInOut}LSU_link.txt", "w") as file:
        file.writelines(lines_link);
    return None

################### Calculate Split KGE and split NSE ################### 
# 
##########################################################################

def fun_kge_s(simulation, observation, dates):
    # Split NSE & KGE
    yrs = dates.year.unique() # Get Years
    nse_s=[]
    kge_s=[]
    for x in yrs:
        ind = dates.year==x # Date Indices of specific year
        # Split NSE
        nse_sb = 1-(np.sum(np.square(simulation[ind]-observation[ind]),axis = 0)/np.sum(np.square(observation[ind]-observation[ind].mean())))
        nse_s.append(nse_sb)
        # Split KGE
        v1 = simulation[ind].mean(axis=0)/observation[ind].mean()
        v2b = (simulation[ind].std(axis=0)/simulation[ind].mean(axis=0))/(observation[ind].std()/observation[ind].mean()) # CV Version 
        v2 = simulation[ind].std(axis=0)/observation[ind].std()
        v3 = np.corrcoef(simulation[ind],observation[ind],rowvar = False)
        v3 = v3[-1,:-1]
 
        kge_sb = 1 - np.sqrt(((v3-1)**2) + ((v2-1)**2) + ((v1-1)**2))
        kge_s.append(kge_sb)
 
    nse_s = np.array(nse_s)
    nse_s = nse_s.mean(axis = 0)
    kge_s = np.array(kge_s)
    kge_s = kge_s.mean(axis = 0)
    return kge_s[0]

def fun_nse_s(simulation, observation, dates):
    # Split NSE & KGE
    yrs = dates.year.unique() # Get Years
    nse_s=[]
    kge_s=[]
    for x in yrs:
        ind = dates.year==x # Date Indices of specific year
        # Split NSE
        nse_sb = 1-(np.sum(np.square(simulation[ind]-observation[ind]),axis = 0)/np.sum(np.square(observation[ind]-observation[ind].mean())))
        nse_s.append(nse_sb)
        # Split KGE
        v1 = simulation[ind].mean(axis=0)/observation[ind].mean()
        v2b = (simulation[ind].std(axis=0)/simulation[ind].mean(axis=0))/(observation[ind].std()/observation[ind].mean()) # CV Version 
        v2 = simulation[ind].std(axis=0)/observation[ind].std()
        v3 = np.corrcoef(simulation[ind],observation[ind],rowvar = False)
        v3 = v3[-1,:-1]
        kge_sb = 1 - np.sqrt(((v3-1)**2) + ((v2-1)**2) + ((v1-1)**2))
        kge_s.append(kge_sb)
 
    nse_s = np.array(nse_s)
    nse_s = nse_s.mean(axis = 0)
    kge_s = np.array(kge_s)
    kge_s = kge_s.mean(axis = 0)
    return nse_s 