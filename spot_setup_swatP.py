"""
Structure from: 
Tobias Houska (2015)
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

Edited and extended by: 
Lucas Alcamo 
This file integrates SWAT+ model in SPOTPY

Created on: 31.08.2023

Procedure:
    - Check that calibration.cal is selected in file.cio
    - Check that relevant outputs are selected in print.prt
    - Save observation data under 'D:/spotpy_SWATp/obs_data/' as 'obs_data.csv'. Beware of format
    - Select path of model and .exe file name 
    - Run using "spot_setup(spotpy.objectivefunctions.[name_obj_function], [mpi-True/False])
"""

import os
from spotpy.parameter import Uniform
import pandas as pd
import spotpy
import sys  
sys.path.insert(0, '../swatPPspotpy/')
from spotpy_functions import * #

class spot_setup(object):

    def __init__(self, df_params, df_variables, par_dict, model_name, flag_ts='month', obj_func=None, mpi=False, name_TxtInOut='TxtInOut'):
        
        '''
        Initializing setup
        df_params     - Parameters (consider structure!)
        df_variables  - Parameters (consider structure!)
        par_dict      - parent directory (consider structure!)
        obj_func      - Objective function 
        mpi           - Parellel Processing; False (default; no parallel processing), if True, parallel processing possible (copies of TxtInOut folders need to be deleted manually!)
        name_TxtInOut - Name of TxtInOut folder
        '''

        # For running the SWAT+ model
        self.name_exe      = 'Rev_61_0_64rel.exe'
        self.name_HRUshp   = 'hrus2.shp'                  # Actual HRUs 
        self.name_LSUshp   = 'lsus2.shp'                  # Actual LSUs
        self.df_params     = df_params
        self.df_variables  = df_variables
        self.all_sims      = []
        self.calib_sims    = []
        self.weights       = list(df_variables.weight[df_variables.calib==True]) # This can be applied to "def objectivefunction" as it is the same order as the return of the simulation results
        
        # Default parameters 
        self.flag_ts       = flag_ts                      # Model timestep ('day' or 'month')
        self.mpi           = mpi                          # True for parallel processing 
        self.name_TxtInOut = name_TxtInOut
        self.obj_func      = obj_func

        # Dependent paths (usually these don't need to be edited)
        if model_name == None:
            self.model_name = os.listdir(par_dict + 'swat_model/')[0]     # Name of model --> choose firs model in "swat_model" folder
        else:
            self.model_name = model_name
        self.path_model    = par_dict + 'swat_model/' + model_name + '/' # Path to model 
        self.path_obs      = par_dict + 'obs_data/prep/'                 # Path to observations
        self.path_Default  = self.path_model + 'Scenarios/Default/'
        self.path_TxtInOut = self.path_model + 'Scenarios/Default/'+self.name_TxtInOut+'/'
        self.path_shapes   = self.path_model + 'Watershed/Shapes/'
          
        # Add parameters to spotpy
        self.params = []
        for ind in df_params.index:
            if pd.isna(df_params.at[ind,'upper_bound']) == False:
                self.params.append(Uniform(name=df_params.at[ind,'name_spotpy'], low=df_params.at[ind,'lower_bound'],high=df_params.at[ind,'upper_bound']))

        # Get simulation dates
        self.date = get_sim_dates(self.path_TxtInOut, self.flag_ts)
        
        # Load Observation data from file (file is located and named specifically)  
        self.trueObs, self.origObs = return_obs_data(self.df_variables, self.path_obs, self.date)

        # Write empty calibration.cal file 
        write_empty_cal_file(self.path_shapes, self.name_HRUshp, self.path_TxtInOut, self.df_params)

        # Create LSU link file (for use in retrieving simulation results)
        write_LSU_link_file(self.path_shapes, self.name_LSUshp, self.path_TxtInOut)
                    
    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, parameter_values):
        
        # Prep for parallel processing (define with which path to run, i.e. which TxtInOut folder)
        self.path_run = prep_parallel_processing(self.mpi, self.path_Default, self.path_TxtInOut, self.name_TxtInOut)

        # Here the model is startet with a given paramter combination
        run_swatP(self.path_run, self.name_exe, parameter_values)

        # Read simulation resuls
        sim_calib, sim_cv = return_sim_data(self.df_variables, self.path_run)
        
        # Add to all_sims (Output of all simulations)      
        if self.mpi == False:
            self.all_sims.append(sim_cv)
            self.calib_sims.append(sim_calib)
        
        # Return output from model
        return sim_calib

    def evaluation(self):
        return self.trueObs

    def objectivefunction(self, simulation, evaluation, params=None):
        
        # If no objective function defined use RMSE
        if not self.obj_func:
            self.obj_func = spotpy.objectivefunctions.rmse

        # Calculate all listed goodness of fit criteria (including the defined!)
        loc_weight       = 0
        
        like             = 0 
        pbias            = 0 
        nashsutcliffe    = 0
        lognashsutcliffe = 0
        rmse             = 0 
        mae              = 0 
        kge              = 0 
        agreement_ind    = 0
        kge_s            = 0
        nse_s            = 0
        
        # If several variables generated use mean of all objective functions
        if type(simulation[0])==list:
            for sim, eva in zip(simulation, evaluation):
                # Defined obj function (this will be used for optimization)
                if str(self.obj_func).split()[1] == 'kge_s':
                    like         += self.obj_func(eva, sim, self.date)*self.weights[loc_weight]/sum(self.weights)
                else:
                    like         += self.obj_func(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                # Other obj functions
                pbias            += spotpy.objectivefunctions.pbias(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                nashsutcliffe    += spotpy.objectivefunctions.nashsutcliffe(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                lognashsutcliffe += spotpy.objectivefunctions.lognashsutcliffe(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                rmse             += spotpy.objectivefunctions.rmse(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                mae              += spotpy.objectivefunctions.mae(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                kge              += spotpy.objectivefunctions.kge(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                agreement_ind    += spotpy.objectivefunctions.agreementindex(eva, sim)*self.weights[loc_weight]/sum(self.weights)
                kge_s            += spotpy.objectivefunctions.kge_s(eva, sim, self.date)*self.weights[loc_weight]/sum(self.weights)
                # kge_s            += fun_kge_s(np.asarray(sim), np.asarray(eva), self.date)*self.weights[loc_weight]/sum(self.weights)
                nse_s            += fun_nse_s(np.asarray(sim), np.asarray(eva), self.date)*self.weights[loc_weight]/sum(self.weights)

                loc_weight += 1
            
        # Otherwise simply calculate all objective functions
        else:
            # Defined obj function (this will be used for optimization)
            if str(self.obj_func).split()[1] == 'kge_s':
                like         = self.obj_func(evaluation, simulation, self.date)
            else:
                like         = self.obj_func(evaluation, simulation)
            # Other obj functions
            pbias            = spotpy.objectivefunctions.pbias(evaluation, simulation)
            nashsutcliffe    = spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)
            lognashsutcliffe = spotpy.objectivefunctions.lognashsutcliffe(evaluation, simulation)
            rmse             = spotpy.objectivefunctions.rmse(evaluation, simulation)
            mae              = spotpy.objectivefunctions.mae(evaluation, simulation)
            kge              = spotpy.objectivefunctions.kge(evaluation, simulation)
            agreement_ind    = spotpy.objectivefunctions.agreementindex(evaluation, simulation)
            kge_s            = spotpy.objectivefunctions.kge_s(evaluation, simulation, self.date)
            # kge_s            = fun_kge_s(np.asarray(simulation), np.asarray(evaluation), self.date)
            nse_s            = fun_nse_s(np.asarray(simulation), np.asarray(evaluation), self.date)

        likes = [like, pbias, nashsutcliffe, lognashsutcliffe, rmse, mae, kge, agreement_ind, kge_s, nse_s]
            
        return likes

