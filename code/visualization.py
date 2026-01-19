import configuration_settings as cs
import data_processing as dp
from colorama import Fore
import sys
import os
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import seaborn as sns

#So i want to create a function for the dynamic approach


def dynamic_t(directory, filename, extension=""):
    #I am adding the ../ in order to create the filepath
    directory = f"../{directory}"
    # Check if the chosen folder exists else terminate

    try:
        os.makedirs(directory, exist_ok= True)
    except OSError:
        print(f" {Fore.RED} Wrong directory name!.{Fore.RESET}")
        sys.exit()

    # Ensure extension has a dot
    if not extension.startswith("."):
        extension = f".{extension}"
        
    output_path = os.path.join(directory, f"{filename}{extension}")
    i=0
    while os.path.exists(output_path):
        output_path = os.path.join(directory,f"{filename}_{i}{extension}")
        i=i+1
    return output_path.replace("\\", "/")   
 


    
    
 