"""
Structure from: 
Tobias Houska (2015)
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

Edited and extended by: 
Lucas Alcamo 
This file integrates SWAT+ model in SPOTPY


Created on: 31.08.2023

Proceudure:
    - Save observation data under 'D:/spotpy_SWATp/obs_data/' as 'obs_data.csv'. Beware of format!
    - Select path of model and .exe file name 
    - Run using "spot_setup(spotpy.objectivefunctions.[name_obj_function])
"""

import os

from spotpy.objectivefunctions import rmse
from spotpy.parameter import Uniform
import pandas as pd
import spotpy
import geopandas as gpd

import sys  
sys.path.insert(0, 'D:/spotpy_SWATp/python_files/')
from spotpy_functions import run_swatP

class spot_setup(object):

    def __init__(self, obj_func=None, mpi=False):
        
        '''
        Initializing setup
        mpi = False (default; no parallel processing), if True, parallel processing possible (copies of TxtInOut folders need to be deleted manually!)
        '''
        
        # For running the SWAT+ model
        self.channel_id    = 1
        self.name_exe      = 'rev60.5.7_64rel.exe'
        self.path_model    = 'D:/SWAT_models/KD_upsdtream/KD_upstream/'
        self.path_obs      = 'D:/spotpy_SWATp/obs_data/'
        self.name_obs      = 'obs_data.csv'
        self.name_HRUshp   = 'hrus2.shp'# Check if 'hrus1.shp' or 'hrus2.shp' is needed
        self.mpi           = mpi 
        
        # Dependent paths
        self.path_Default  = self.path_model + 'Scenarios/Default/'
        self.path_TxtInOut = self.path_model + 'Scenarios/Default/TxtInOut/'
        self.path_shapes   = self.path_model + 'Watershed/Shapes/'
        
        # Set objective functino 
        self.obj_func      = obj_func
        
        # Add parameters
        # Note: - Subbasins to be ginvin in a list
        #       - If several subbasins in one list these will be treated as one parameter
        #       - For variing values for different subbasins repeat the parameter with equivalent 
        #         range and respective subbasin list (even if just one subbasin in the list)
        # 
        #         NAME         CHANGE_TYPE,   LOWER_BOUND     Upper_BOUND   SUBBASIN (if all type 'None')
        data = [['cn2',          'pctchg',      -10,            15,           None     ],
                ['perco',        'abschg',       0.0,           20.0,         None     ],
                ['k',            'pctchg',      -30.0,          30.0,         None     ],
                ['awc',          'pctchg',      -30.0,          20.0,         None     ],
                ['snofall_tmp',  'absval',      -2.0,           5.0,          None     ],  # -5, 5 Maximum Range!
                ['snomelt_tmp',  'absval',       0.0,           5.0,          None     ],  # -5, 5 Maximum Range!
                ['snomelt_max',  'absval',       0.0,           3.0,          None     ],  # 0, 5 Maximum Range!
                ['snomelt_min',  'absval',       0.0,           4.0,          None     ],  # 0, 5 Maximum Range!
                ['snomelt_lag',  'absval',       0.0,           1.0,          None     ],  # 0, 1 Maximum Range!
                ['surlag',       'absval',       0.05,          20,           None     ]]  # 0.05, 24 Maximum Range!

        df = pd.DataFrame(data, columns=['name', 'change_type', 'lower_bound', 'upper_bound', 'subbasins'])
        
        # Edit names for spotpy
        for i in df.index:
            suff      = ''
            subbasins = df.at[i, 'subbasins']
            if subbasins!= None:
                # suff = '_suff_sub' # Add '_suff_' as an indicator for a suffix which can be deleted later on when writing calibration file
                for j in subbasins:
                    suff += '_sub' + str(j)

            df.at[i,'name_spotpy'] = df.at[i,'name'] + suff
        
        # Add parameters to spotpy
        self.params = []
        for ind in df.index:
            self.params.append(Uniform(name=df.at[ind,'name_spotpy'], low=df.at[ind,'lower_bound'],high=df.at[ind,'upper_bound']))

        # Add observations here
        #             !!! SWAT model has to be run at least once with the given timeframe before starting this code !!!
        self.date = [] 
        with open(f"{self.path_TxtInOut}channel_sd_mon.txt", "r") as file:
            for _ in range(3):
                next(file)
            for line in file:
                columns = line.split()  # Split the line into columns
                unit    = int(columns[4]) 
                if unit == 1:
                    day   = '01'
                    month = columns[1]
                    year  = columns[3]
                    if int(month) <= 9:
                        month = '0'+ month 
                    date_complete = day + '.' + month + '.' + year
                    self.date.append(date_complete) 

        # Load Observation data from file (file is located and named specifically)
        flo_obs      = pd.read_csv(self.path_obs + self.name_obs, index_col='Date', parse_dates=True, date_format='%d/%m/%Y',na_values=[-9999.0])
        self.trueObs = []
        flo_obs      = flo_obs.loc[pd.to_datetime(self.date[0], dayfirst=True):pd.to_datetime(self.date[-1], dayfirst=True)]
        self.trueObs = list(flo_obs['Data'])
        
        ##################### Initiate calibration.cal file #####################
        #
        # Read LSUs file to link subbasin numbers with LSUIDs
        HRU_file = gpd.read_file(self.path_shapes + self.name_HRUshp)

        # Write blank calibration.cal file
        header_line = 'calibration.cal: written by SWAT++SPOTPY\n'
        second_line = f"{str(len(df.index))}\n"
        column_line = 'cal_parm               chg_typ       chg_val     conds  soil_lyr1  soil_lyr2       yr1       yr2      day1      day2   obj_tot\n'
        lines       = [header_line,second_line,column_line]
        
        # For HRU link file
        header_link = 'HRU_link: written by SWAT++SPOTPY\n'
        lines_link  = [header_link]
        
 
        for ind in df.index:
            # Define parameters
            name_var       = df.at[ind,'name']
            name_spotpy    = df.at[ind,'name_spotpy']
            chg_typ        = df.at[ind,'change_type']
            subbasins_list = df.at[ind,'subbasins']
            obj_tot        = 0 
            HRU_range      = ''

            # If subbasins were defined
            if subbasins_list != None:

                for subbasin_nr in subbasins_list:
                    
                    # Range of HRUs within the subbasin
                    HRUs_list = list(HRU_file[HRU_file['Subbasin'] == subbasin_nr]['HRUS'].values)
                    HRUs_list.sort();
                    HRU_range = str(min(HRUs_list)) + '  -' + str(max(HRUs_list))
                    
                    # Object total is 2 when subbasin selected (range) 
                    obj_tot   = 2
                    
                    # Add line for each subbasin and parameter 
                    line = f"{name_var:20}{chg_typ:>10}{'replace':>14}{'0':>10}{'0':>11}{'0':>11}{'0':>10}{'0':>10}{'0':>10}{'0':>10}{obj_tot:>10}  {HRU_range}\n"
                    lines.append(line);
                    
                    # Write lines to link subbasins to HRU ranges
                    line_link = f'{name_var} {HRU_range} {name_spotpy}\n'
                    lines_link.append(line_link)
                    
            else:
                # Add one line for each parameter if no subbains were defined
                line = f"{name_var:20}{chg_typ:>10}{'replace':>14}{'0':>10}{'0':>11}{'0':>11}{'0':>10}{'0':>10}{'0':>10}{'0':>10}{obj_tot:>10}\n"
                lines.append(line);            
        
        # Write blank calibration file 
        with open(f"{self.path_TxtInOut}calibration_read.cal", "w") as file:
            file.writelines(lines);
            
        # Write file to link subbasins to HRU ranges
        with open(f"{self.path_TxtInOut}HRU_link.txt", "w") as file:
            file.writelines(lines_link);
            
                    
    def parameters(self):
        return spotpy.parameter.generate(self.params)


    def simulation(self, parameter_values):
        
        # Prep for parallel processing 
        if self.mpi == True:
            #OS must be windows
            comm = MPI.COMM_WORLD 
            core_nr = str(comm.Get_rank()+1) # +1 to prevent zero
            # Create a path and copy TxtInOut for each new core_nr seen
            path_core = self.path_Default + '/TxtInOut_core' + str(core_nr)
            # Copy TxtInOut folder if it doesn't exist!
            if os.path.exists(path_core + os.sep) == False:
                try:
                    shutil.copytree(self.path_Default + '/TxtInOut', path_core)
                except WindowsError as e:
                    print ("ERROR: WINDOWSERROR = ",e)
                except :
                    print ("ERROR: Some other error happened")   
            # Define from which path to run 
            self.path_run = path_core + '/'
            
        else: 
            # Define from which path to run 
            self.path_run = self.path_TxtInOut
            
        # Here the model is actualy startet with one paramter combination
        data = run_swatP(self.path_run, self.name_exe, parameter_values, self.channel_id)
            
        sim = []
        sim = data
        
        # Return output from model
        return sim

    def evaluation(self):
        return self.trueObs

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like
