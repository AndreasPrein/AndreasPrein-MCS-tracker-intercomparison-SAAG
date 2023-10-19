#!/usr/bin/env python
# coding: utf-8

# # Mesoscale Convective System (MCS) Feature Tracking 

# This is a working example of an MCS feature tracking algorithm that uses hourly precipitation and brithness temperature data to classify MCSs. Earlier versions of this tracking algorithm have been used in [Poujol et al. (2020)](https://link.springer.com/article/10.1007/s00382-020-05466-1) and [Prein et al. (2021) ](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0546).
# 
# The tracking algorithm contains three main steps:
# 
# 
# *   Create a mask array containing zeros (no object) and ones (object) by thresholding the precipitation and brightness temperature field (Tb) field.
# *   Identify connected features in the masked field and label them with a unique index. Grid cells are connected if they are adjacent in space and time (diagonal connections are allowed). 
# *   Analyze each identified feateture if it fulfills the minimum requirements for MCSs that are provided by the user.
# 
# The mimimum requirements are associated with a minimum threshold that the feature must exceed (e.g., 2 mm/h precipitation rates), minimal maximum values (e.g. peak hourly rain rates), minimum size of the object, and a minimum duration for which the maximum value and object size must be exceeded.
# 
# The current example uses [GPM-IMERG v6](https://gpm.nasa.gov/data/imerg) precipitation and [MERGIR](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary) Tb observations for MCS tracking for June 1, 2016 over the Contigeous United States. Both datasets are averaged to hourly values (24-time slizes) and the 4 km MERGIR data is regridded to the GPM-IMERG grid. The resulting data is stored in the "DATA_all" numpy array which has the dimensions [time, latitude, longitude, variables]. This array is passed to the "MCStracking" function, which performs the tracking. A few example outputs from this function are visualized at the end of the notebook.
# 
# The "DATA_all" function returns a directory and a matrix and writes a netcdf file.
# 
# 
# *   The "grMCSs" directory contains sub-directories for each MCS that was identified. Each of the sub-directories contains MCS characteristics such as the track, speed, or max. precipitation. We plot the statistics of this directory at the end of this notebook.
# *   The "MCS_obj" variable is a nupy array that conains the labels of tracked MCSs. Each MCS has a unique index and non-MCS areas are set to zero. The dimensions of "MCS_obj" are [time, lat, lon]. This variable can also be found in the NetCDF file that is described below. 
# *   The "20160701_CONUS-MCS-tracking.nc" file contains the original precipitation and Tb data that was used for the MCS detection and the labels of each feature that was identified. This file will give you a good overview how the MCSs look like and you can use its content if you would like to perform you own statistics on the data.
# 
# 
# 
# If users whant to use a different dataset for MCS tracking (e.g., model results) they have to read in and process the precipitation data into the same format as the "DATA_all" data.
# 
# Please send questions or comments concerning this code to Andreas Prein (prein@ucar.edu).
# 
# 
# 
# 

# In[1]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
import matplotlib.path as mplPath
import sys
from calendar import monthrange
import warnings
warnings.filterwarnings("ignore")
from itertools import groupby
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from pylab import *
import h5py as h5py
import cartopy
import cartopy.crs as ccrs
from tqdm import tqdm
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from skimage.measure import regionprops
import time
from tqdm import tqdm

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import h5py
import xarray as xr
import netCDF4


# In[2]:


import mcs_config as cfg

# #### Functions from "Tracking_Function.py" file
# from Tracking_Functions import ObjectCharacteristics
from Tracking_Functions import interp_weights
from Tracking_Functions import interpolate
from Tracking_Functions import detect_local_minima
from Tracking_Functions import Feature_Calculation
from Tracking_Functions import haversine
from Tracking_Functions import BreakupObjects
from Tracking_Functions import ConnectLon
from Tracking_Functions import ReadERA5
from Tracking_Functions import minimum_bounding_rectangle 
from Tracking_Functions import is_land 
from Tracking_Functions import DistanceCoord 
from Tracking_Functions import MultiObjectIdentification
from Tracking_Functions import MCStracking
from Tracking_Functions import calc_object_characteristics
from Tracking_Functions import calc_grid_distance_area
from Tracking_Functions import calculate_area_objects
from Tracking_Functions import remove_small_short_objects
from Tracking_Functions import clean_up_objects


# In[3]:


###  USER MODIFY SECTION
DataName = str(sys.argv[1]) #'IMERG-MERGIR' # can be ['IMERG-MERGIR', 'SAAG-WRF4']
FocusRegion = [20, -30, -60, -85] #  N, E, S, W
WaterYear = str(sys.argv[2]) #'WY2016' # can be [WY2011, WY2016, WY2019]

Variables = ['PR','Tb']
DataOutDir = '/glade/campaign/mmm/c3we/prein/Papers/2022-SA-MCS-Tracking/data/My-Track-Results/tests2/'
if not os.path.exists(DataOutDir):
    os.makedirs(DataOutDir)

# target grid 
sTarGrid ='IMERG'         # name of target grid
dT = 1                    # temporal resolution of data for tracking in hours

# # MINIMUM REQUIREMENTS FOR FEATURE DETECTION
# # precipitation tracking options
# SmoothSigmaP = 0          # Gaussion std for precipitation smoothing
# Pthreshold = 5            # precipitation threshold [mm/h]
# MinTimePR = 4             # minum lifetime of PR feature in hours
# MinAreaPR = 20000          # minimum area of precipitation feature in km2

# # Brightness temperature (Tb) tracking setup
# SmoothSigmaC = 0          # Gaussion std for Tb smoothing
# Cthreshold = 241          # minimum Tb of cloud shield
# MinTimeC = 4             # minium lifetime of cloud shield in hours
# MinAreaC = 40000          # minimum area of cloud shield in km2

# # MCs detection
# MCS_Minsize = MinAreaPR   # minimum area of MCS precipitation object in km2
# MCS_minPR = 10            # minimum max precipitation in mm/h
# MCS_MinPeakPR = 10        # Minimum lifetime peak of MCS precipitation
# CL_MaxT = 225             # minimum brightness temperature
# CL_Area = MinAreaC        # min cloud area size in km2
# MCS_minTime = 1           # minimum lifetime of MCS


# In[4]:


### SET DATA SPECIFIC VARIABLES   
if WaterYear == 'WY2011':
    StartDay = datetime.datetime(2010, 6, 1,0)
    StopDay = datetime.datetime(2011, 5, 31,23)    
#     StartDay = datetime.datetime(2011, 1, 1,0)
#     StopDay = datetime.datetime(2011, 1, 31,23)    
elif WaterYear == 'WY2016':
    StartDay = datetime.datetime(2015, 6, 1,0)
    StopDay = datetime.datetime(2016, 5, 31,23)    
elif WaterYear == 'WY2019':
    StartDay = datetime.datetime(2018, 6, 1,0)
    StopDay = datetime.datetime(2019, 5, 31,23)


# In[5]:


# ===============================================
# Setup variables for computation

TimeHH=pd.date_range(StartDay, end=StopDay, freq='h')
TimeMM=pd.date_range(StartDay, end=StopDay + datetime.timedelta(days=1), freq='m')
TimeDD = pd.date_range(StartDay, end=StopDay, freq='d')
TimeBT = pd.date_range(StartDay, end=StopDay, freq='h')
Time = pd.date_range(StartDay, end=StopDay, freq=str(dT)+'h')
Years = np.unique(TimeMM.year)
iHHall = np.array(range(len(TimeHH)))

# # ### Read the target grid
ncid=Dataset('/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/GPM/2001/merg_2001123110_4km-pixel.nc', mode='r')
Lat=np.squeeze(ncid.variables['lat'][:])
Lon=np.squeeze(ncid.variables['lon'][:])
ncid.close()
Lon,Lat = np.meshgrid(Lon,Lat)


if Lon.max() > 180:
    Lon[Lon > 180] = Lon[Lon > 180]-360

if (FocusRegion[1] > 0) & (FocusRegion[3] < 0):
    # region crosses zero meridian
    iRoll = np.sum(Lon[0,:] < 0)
else:
    iRoll=0
Lon = np.roll(Lon,iRoll, axis=1)

iNorth = np.argmin(np.abs(Lat[:,0] - FocusRegion[0]))+1
iSouth = np.argmin(np.abs(Lat[:,0] - FocusRegion[2]))
iEast = np.argmin(np.abs(Lon[0,:] - FocusRegion[1]))+1
iWest = np.argmin(np.abs(Lon[0,:] - FocusRegion[3]))
print(iNorth,iSouth,iWest,iEast)

Lon = Lon[iSouth:iNorth,iWest:iEast]
Lat = Lat[iSouth:iNorth,iWest:iEast]

Mask = np.copy(Lon); Mask[:]=1
DATA_all = np.zeros((len(Time),Lon.shape[0],Lon.shape[1],len(Variables))); DATA_all[:] = np.nan


# ### Read Data

# In[6]:


for hh in tqdm(range(len(Time))):
    TimeAct = Time[hh]
    YYYY = str(TimeAct.year)
    MM   = str(TimeAct.month).zfill(2)
    DD   = str(TimeAct.day).zfill(2)
    HH   = str(TimeAct.hour).zfill(2)
    if DataName == 'IMERG-MERGIR':
        NcVARS = ['precipitationCal','Tb']
        sFileAct = '/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/GPM/'+str(TimeAct.year)+'/merg_'+YYYY+MM+DD+HH+'_4km-pixel.nc'
    elif DataName == 'SAAG-WRF4':
        NcVARS = ['rainrate','tb']
        sFileAct = '/glade/campaign/mmm/c3we/prein/SouthAmerica/MCS-Tracking/'+WaterYear+'/WRF/tb_rainrate_'+YYYY+'-'+MM+'-'+DD+'_'+HH+':00.nc'
        
    ncid=Dataset(sFileAct, mode='r')
    PR_act = np.squeeze(ncid.variables[NcVARS[0]][:])
    Tb_act = np.squeeze(ncid.variables[NcVARS[1]][:])
    ncid.close()
    
    if DataName == 'IMERG-MERGIR':
        PR_act = PR_act[0,:,:]
        Tb_act1 = np.copy(Tb_act[0,:,:])
        # try to fill up nan values with data from next 30 min obs.
        NAN = Tb_act[0,:,:].mask == True
        Tb_act1[NAN] = Tb_act[1,NAN]
        Tb_act = Tb_act1
    
    DATA_all[hh,:,:,Variables.index('PR')] = PR_act
    DATA_all[hh,:,:,Variables.index('Tb')] = Tb_act

# Mask out the focus domain if needed
DATA_all[:,Mask == 0,:] = np.nan


# ### Perform MCS Tracking

# In[7]:


# NCfile = DataOutDir+WaterYear+'_'+DataName+'_Prein_SAAG-MCS-tracking_Smooth.nc'

# grMCSs, MCS_obj = MCStracking(DATA_all[:,:,:,Variables.index('PR')],
#                               DATA_all[:,:,:,Variables.index('Tb')],
#                               Time,
#                               Lon,
#                               Lat,
#                               NCfile,
#                               DataOutDir,
#                               DataName)


# In[ ]:


# def MCStracking(
#     pr_data,
#     bt_data,
#     times,
#     Lon,
#     Lat,
#     nc_file):
NCfile = DataOutDir+WaterYear+'_'+DataName+'_Prein_SAAG-MCS-tracking_Smooth.nc'

pr_data = DATA_all[:,:,:,Variables.index('PR')]
bt_data = DATA_all[:,:,:,Variables.index('Tb')]
times = Time
Lon = Lon
Lat = Lat
nc_file = NCfile
DataOutDir = DataOutDir
data_name = WaterYear+'_'+DataName+'_Prein_SAAG-MCS-tracking.nc'


import mcs_config as cfg
from skimage.measure import regionprops
start_time = time.time()
#Reading tracking parameters

DT = cfg.DT

#Precipitation tracking setup
smooth_sigma_pr = 0 # cfg.smooth_sigma_pr   # [0] Gaussion std for precipitation smoothing
thres_pr        = 1 # cfg.thres_pr     # [2] precipitation threshold [mm/h]
min_time_pr     = 1 # cfg.min_time_pr     # [3] minum lifetime of PR feature in hours
min_area_pr     = 20000 #      # [5000] minimum area of precipitation feature in km2
# Brightness temperature (Tb) tracking setup
smooth_sigma_bt = 0   # cfg.smooth_sigma_bt   #  [0] Gaussion std for Tb smoothing
thres_bt        = 241 # cfg.thres_bt     # [241] minimum Tb of cloud shield
min_time_bt     = 4   # cfg.min_time_bt       # [9] minium lifetime of cloud shield in hours
min_area_bt     = 40000 # cfg.min_area_bt       # [40000] minimum area of cloud shield in km2
bt_overshoot    = 225 # K
# MCs detection
MCS_min_pr_MajorAxLen  = 100 # cfg.MCS_min_pr_MajorAxLen    # [100] km | minimum length of major axis of precipitation object
MCS_thres_pr       = 1 #cfg.MCS_thres_pr      # [10] minimum max precipitation in mm/h
MCS_thres_peak_pr   = 10 #cfg.MCS_thres_peak_pr  # [10] Minimum lifetime peak of MCS precipitation
MCS_thres_bt     = thres_bt #cfg.MCS_thres_bt        # [225] minimum brightness temperature
MCS_min_area_bt         = min_area_bt #cfg.MCS_min_area_bt        # [40000] min cloud area size in km2
MCS_min_time     = 4 #cfg.MCS_min_time    # [4] minimum time step


#     DT = 1                    # temporal resolution of data for tracking in hours

#     # MINIMUM REQUIREMENTS FOR FEATURE DETECTION
#     # precipitation tracking options
#     smooth_sigma_pr = 0          # Gaussion std for precipitation smoothing
#     thres_pr = 2            # precipitation threshold [mm/h]
#     min_time_pr = 3             # minum lifetime of PR feature in hours
#     min_area_pr = 5000          # minimum area of precipitation feature in km2

#     # Brightness temperature (Tb) tracking setup
#     smooth_sigma_bt = 0          # Gaussion std for Tb smoothing
#     thres_bt = 241          # minimum Tb of cloud shield
#     min_time_bt = 9              # minium lifetime of cloud shield in hours
#     min_area_bt = 40000          # minimum area of cloud shield in km2

#     # MCs detection
#     MCS_min_area = min_area_pr   # minimum area of MCS precipitation object in km2
#     MCS_thres_pr = 10            # minimum max precipitation in mm/h
#     MCS_thres_peak_pr = 10        # Minimum lifetime peak of MCS precipitation
#     MCS_thres_bt = 225             # minimum brightness temperature
#     MCS_min_area_bt = MinAreaC        # min cloud area size in km2
#     MCS_min_time = 4           # minimum lifetime of MCS

#Calculating grid distances and areas

_,_,grid_cell_area,grid_spacing = calc_grid_distance_area(Lat,Lon)
grid_cell_area[grid_cell_area < 0] = 0

obj_structure_3D = np.ones((3,3,3))

start_day = times[0]


# connect over date line?
crosses_dateline = False
if (Lon[0, 0] < -176) & (Lon[0, -1] > 176):
    crosses_dateline = True

end_time = time.time()
print(f"======> 'Initialize MCS tracking function: {(end_time-start_time):.2f} seconds \n")
start_time = time.time()

# # --------------------------------------------------------
# # TRACKING PRECIP OBJECTS
# # --------------------------------------------------------
# print("        track  precipitation")

# pr_smooth= filters.gaussian_filter(
#     pr_data, sigma=(0, smooth_sigma_pr, smooth_sigma_pr)
# )
# pr_mask = pr_smooth >= thres_pr * DT
# objects_id_pr, num_objects = ndimage.label(pr_mask, structure=obj_structure_3D)
# print("            " + str(num_objects) + " precipitation object found")

# # connect objects over date line
# if crosses_dateline:
#     objects_id_pr = ConnectLon(objects_id_pr)

# # get indices of object to reduce memory requirements during manipulation
# object_indices = ndimage.find_objects(objects_id_pr)


# #Calcualte area of objects
# area_objects = calculate_area_objects(objects_id_pr,object_indices,grid_cell_area)

# # Keep only large and long enough objects
# # Remove objects that are too small or short lived
# pr_objects = remove_small_short_objects(objects_id_pr,area_objects,min_area_pr,min_time_pr,DT)

# grPRs = calc_object_characteristics(
#     pr_objects,  # feature object file
#     pr_data,  # original file used for feature detection
#     DataOutDir+DataName+"_PR_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
#     times,  # timesteps of the data
#     Lat,  # 2D latidudes
#     Lon,  # 2D Longitudes
#     grid_spacing,
#     grid_cell_area,
#     min_tsteps=int(min_time_pr/ DT), # minimum lifetime in data timesteps
# )

# end_time = time.time()
# print(f"======> 'Tracking precip: {(end_time-start_time):.2f} seconds \n")
# start_time = time.time()

# --------------------------------------------------------
# TRACKING CLOUD (BT) OBJECTS
# --------------------------------------------------------
print("            track  clouds")
bt_smooth = filters.gaussian_filter(
    bt_data, sigma=(0, smooth_sigma_bt, smooth_sigma_bt)
)
bt_mask = bt_smooth <= thres_bt
objects_id_bt, num_objects = ndimage.label(bt_mask, structure=obj_structure_3D)
print("            " + str(num_objects) + " cloud object found")

# connect objects over date line
if crosses_dateline:
    print("            connect cloud objects over date line")
    objects_id_bt = ConnectLon(objects_id_bt)

# get indices of object to reduce memory requirements during manipulation
object_indices = ndimage.find_objects(objects_id_bt)

#Calcualte area of objects
area_objects = calculate_area_objects(objects_id_bt,object_indices,grid_cell_area)

# Keep only large and long enough objects
# Remove objects that are too small or short lived
objects_id_bt = remove_small_short_objects(objects_id_bt,area_objects,min_area_bt,min_time_bt,DT)

end_time = time.time()
print(f"======> 'Tracking clouds: {(end_time-start_time):.2f} seconds \n")
start_time = time.time()

print("            break up long living cloud shield objects that have many elements")
objects_id_bt = BreakupObjects(objects_id_bt, int(min_time_bt / DT), DT)

end_time = time.time()
print(f"======> 'Breaking up cloud objects: {(end_time-start_time):.2f} seconds \n")
start_time = time.time()

# grCs = calc_object_characteristics(
#     objects_id_bt,  # feature object file
#     bt_data,  # original file used for feature detection
#     DataOutDir+DataName+"_BT_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
#     times,  # timesteps of the data
#     Lat,  # 2D latidudes
#     Lon,  # 2D Longitudes
#     grid_spacing,
#     grid_cell_area,
#     min_tsteps=int(min_time_bt / DT), # minimum lifetime in data timesteps
# 

objects_id_bt = clean_up_objects(objects_id_bt,
                                 DT,
                                 min_tsteps=int(MCS_min_time/DT))   


end_time = time.time()
# --------------------------------------------------------
# CHECK IF PR OBJECTS QUALIFY AS MCS
# (or selected strom type according to msc_config.py)
# --------------------------------------------------------
print(f"======> 'check if Tb objects quallify as MCS (or selected storm type)")
start_time = time.time()
# check if precipitation object is from an MCS
object_indices = ndimage.find_objects(objects_id_bt)
MCS_objects = np.zeros(objects_id_bt.shape,dtype=int)

for iobj,_ in tqdm(enumerate(object_indices)):
    if object_indices[iobj] is None:
        continue

    time_slice = object_indices[iobj][0]
    lat_slice  = object_indices[iobj][1]
    lon_slice  = object_indices[iobj][2]


    tb_object_slice= objects_id_bt[object_indices[iobj]]
    tb_object_act = np.where(tb_object_slice==iobj+1,True,False)
    if len(tb_object_act) < min_time_bt:
        continue
    
    tb_slice =  bt_data[object_indices[iobj]]
    tb_act = np.copy(tb_slice)
    tb_act[~tb_object_act] = np.nan

    bt_object_slice = objects_id_bt[object_indices[iobj]]
    bt_object_act = np.copy(bt_object_slice)
    bt_object_act[~tb_object_act] = 0

    area_act = np.tile(grid_cell_area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
    area_act[~tb_object_act] = 0

    ### Calculate cloud properties
    tb_size = np.array(np.sum(area_act,axis=(1,2)))
    tb_min = np.array(np.nanmin(tb_act,axis=(1,2)))

    ### Calculate precipitation properties
    pr_act = np.copy(pr_data[object_indices[iobj]])
    pr_act[tb_object_act == 0] = np.nan
    
    pr_peak_act = np.array(np.nanmax(pr_act,axis=(1,2)))
    
    pr_region_act = pr_act >= MCS_thres_pr
    area_act = np.tile(grid_cell_area[lat_slice, lon_slice], (tb_act.shape[0], 1, 1))
    area_act[~pr_region_act] = 0
    pr_under_cloud = np.array(np.sum(area_act,axis=(1,2)))/1000**2 
    
    
    # Test if object classifies as MCS
    tb_size_test = np.max(np.convolve((tb_size / 1000**2 >= min_area_bt), np.ones(min_time_bt), 'valid') / min_time_bt) == 1
    tb_overshoot_test = np.max((tb_min  <= bt_overshoot )) == 1
    pr_peak_test = np.max(np.convolve((pr_peak_act >= MCS_thres_peak_pr ), np.ones(MCS_min_time), 'valid') / MCS_min_time) ==1
    pr_area_test = np.max((pr_under_cloud >= min_area_pr)) == 1

    MCS_test = (
                tb_size_test
                & tb_overshoot_test
                & pr_peak_test
                & pr_area_test
    )

    # assign unique object numbers
    tb_object_act = np.array(tb_object_act).astype(int)
    tb_object_act[tb_object_act == 1] = iobj + 1

    window_length = int(MCS_min_time / DT)
    moving_averages = np.convolve(MCS_test, np.ones(window_length), 'valid') / window_length

#     if iobj+1 == 19:
#         stop()

    if MCS_test == 1:
        TMP = np.copy(MCS_objects[object_indices[iobj]])
        TMP = TMP + tb_object_act
        MCS_objects[object_indices[iobj]] = TMP

    else:
        continue
end_time = time.time()
print(f"======> 'Calculate cloud characteristics: {(end_time-start_time):.2f} seconds \n")
start_time = time.time()

MCS_objects = clean_up_objects(MCS_objects,
                                   DT,
                                    min_tsteps=int(MCS_min_time/DT))   


#if len(objects_overlap)>1: import pdb; pdb.set_trace()
# objects_id_MCS, num_objects = ndimage.label(MCS_objects, structure=obj_structure_3D)
grMCSs = calc_object_characteristics(
    MCS_objects,  # feature object file
    pr_data,  # original file used for feature detection
    DataOutDir+DataName+"_MCS_"+str(start_day.year)+str(start_day.month).zfill(2)+'.pkl',
    times,  # timesteps of the data
    Lat,  # 2D latidudes
    Lon,  # 2D Longitudes
    grid_spacing,
    grid_cell_area,
    min_tsteps=int(MCS_min_time / DT), # minimum lifetime in data timesteps
)

end_time = time.time()
print(f"======> 'MCS tracking: {(end_time-start_time):.2f} seconds \n")
start_time = time.time()


###########################################################
###########################################################
## WRite netCDF with xarray
if nc_file is not None:
    print ('Save objects into a netCDF')

    fino=xr.Dataset({'MCS_objects':(['time','y','x'],MCS_objects),
                     'PR':(['time','y','x'],pr_data),
#                      'PR_objects':(['time','y','x'],objects_id_pr),
                     'BT':(['time','y','x'],bt_data),
                     'BT_objects':(['time','y','x'],objects_id_bt),
                     'lat':(['y','x'],Lat),
                     'lon':(['y','x'],Lon)},
                     coords={'time':times.values})

#     fino.to_netcdf(nc_file,mode='w',encoding={'PR':{'zlib': True,'complevel': 5},
# #                                              'PR_objects':{'zlib': True,'complevel': 5},
#                                              'BT':{'zlib': True,'complevel': 5},
#                                              'BT_objects':{'zlib': True,'complevel': 5},
#                                              'MCS_objects':{'zlib': True,'complevel': 5}})
    fino.to_netcdf(nc_file,mode='w')


# In[ ]:




