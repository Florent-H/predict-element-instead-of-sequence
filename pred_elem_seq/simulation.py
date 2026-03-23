from pred_elem_seq import write_dic_to_file
import pandas as pd
from pathlib import Path
import os
import re
import subprocess
from datetime import datetime
import shutil

class Building(object):
    def __init__(
        self,
        name,
        year,
        sim_time_idx,
        unknown_params,
        var_data,
        obj_names,
        weather_path,
    ):
        self.name = name
        self.year = year
        self.sim_time_idx = sim_time_idx
        self.sim_time_freq = sim_time_idx.freq.freqstr
        self.unknown_params = unknown_params
        self.unknown_param_names = list(self.unknown_params.keys())
        self.n_unknown_params = len(unknown_params)
        self.var_data = var_data
        self.var_names = list(var_data.columns)
        self.n_vars = len(self.var_names)
        self.obj_names = obj_names
        self.n_objs = len(obj_names)
        self.weather_path = weather_path
        self.simulated = None
        # this is the EnergyPlus simulation subdirectory
        self.eprun_dir = None
        # this is the top directory for EnergyPlus simulation subdirectories
        self.eptest_dir = Path("pred_elem_seq/eptest")

    def evaluate(self, param_array=None, weather_path=None):
        # if simulating different weather files
        if weather_path is not None:
            # assign weather path to building attribute
            self.weather_path = weather_path
        # assign unknown parameter values from array to building model
        if param_array is not None:
            # do not include the fixed parameters when assigning the parameter array
            unknown_param_array = param_array[-self.n_unknown_params :]
            for p, param_val in zip(unknown_param_array, self.unknown_params.values()):
                param_val["val"] = p

        # write new parameters to idf file
        idf_path = self.write_unknown_params()
        # run the energyplus engine
        self.run(idf_path)
        # get simulated energy from csv output file
        simulated_energy = self.get_simulated_energy()

        return simulated_energy

    def write_unknown_params(self):
        # make the uniquely-named eprun directory with the copied epw weather file
        self.make_eprun_dir()
        # the idf file is written in the EnergyPlus run directory
        write_idf_path = self.eprun_dir / f"{self.name}.idf"
        template_dir = Path("pred_elem_seq/datafiles/ep/template")
        template_idf_path = template_dir / f"{self.name}.idf"
        # use the %parameter_name% markers in the input file
        self.write_idf_use_markers(template_idf_path, write_idf_path)

        # write parameters dictionary to json file
        parameter_path = self.eprun_dir / f"{self.name}.json"
        write_dic_to_file(self.unknown_params, parameter_path)

        return write_idf_path

    def make_eprun_dir(self):
        current_time = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S-%f")
        # the EnergyPlus run directory is uniquely named to prevent duplicate directory names when running in parallel
        self.eprun_dir = self.eptest_dir / f"run_{current_time}_{hash(self)}"
        # create EnergyPlus run directory
        self.eprun_dir.mkdir()

    def write_idf_use_markers(self, template_path, write_path):
        with open(template_path, "r") as r_f:
            with open(write_path, "w+") as w_f:
                self.write_idf_lines(r_f, w_f)

    def write_idf_lines(self, r_f, w_f):
        for line in r_f:
            for param in self.unknown_param_names:
                if param == "wallIns":
                    ins_rvalue = self.unknown_params[param]["val"] - 2.39
                    ins_rsivalue = ins_rvalue / 5.678
                    ins_thickness = ins_rsivalue * 0.049
                    line = re.sub("%wallInsThc%", str(ins_thickness), line)

                elif param == "roofIns":
                    ins_rvalue = self.unknown_params[param]["val"] - 0.79
                    ins_rsivalue = ins_rvalue / 5.678
                    ins_thickness = ins_rsivalue * 0.049
                    line = re.sub("%roofInsThc%", str(ins_thickness), line)

                elif param == "infRate":
                    inf_val = self.unknown_params[param]["val"] / 1000
                    line = re.sub(f"%{param}%", str(inf_val), line)

                elif param == "TSetback":
                    tset_night = (
                        self.unknown_params["THeat"]["val"]
                        - self.unknown_params[param]["val"]
                    )
                    line = re.sub(f"%{param}%", str(tset_night), line)

                elif param == "TSetup":
                    tset_night = (
                        self.unknown_params["TCool"]["val"]
                        + self.unknown_params[param]["val"]
                    )
                    line = re.sub(f"%{param}%", str(tset_night), line)

                else:
                    line = re.sub(
                        f"%{param}%",
                        str(self.unknown_params[param]["val"]),
                        line,
                    )

            # write the new file line
            w_f.writelines(line)


    def run(self, bem_path):
        ep_run_string = (
            f'{Path("C:/EnergyPlusV24-1-0/energyplus.exe")} '
            + f'--weather "{self.weather_path}" '
            + f'--output-directory "{self.eprun_dir}" '
            + f"--output-prefix {bem_path.stem} "
            + f'"{bem_path}"'
        )
        subprocess.run(ep_run_string, cwd=Path.cwd(), capture_output=True)

    def get_simulated_energy(self):
        # get simulation data by reading csv output file from Energy Plus
        output_path = self.eprun_dir / f"{self.name}out.csv"
        # create data frame out of csv output file of energy plus
        sim_energy_csv = pd.read_csv(output_path, index_col=0)
        # the EnergyPlus outputs align directly with the objectives
        sim_energy_csv.columns = self.obj_names
        # convert Joules to kWh
        sim_energy = sim_energy_csv / 3.6e6
        # set simulated dataframe index to the same one as the measured energy index
        sim_energy.index = self.sim_time_idx
        # write simulated attribute of Building object
        self.simulated = sim_energy

        return sim_energy

def get_parameters(param_path):
    import csv

    param_names = []
    lower_bound = []
    upper_bound = []

    with open(param_path, "r", newline="") as fr:
        reader = csv.reader(fr)
        for line in reader:
            param_names.append(line[0])
            lower_bound.append(float(line[1]))
            upper_bound.append(float(line[2]))
    parameters = {
        key: {
            "val": (lower_bound[i] + upper_bound[i]) / 2,
            "lower_bound": lower_bound[i],
            "upper_bound": upper_bound[i],
        }
        for i, key in enumerate(param_names)
    }
    return parameters


def clear_sim_folders():
    # clear folders that have accumulated simulation output files
    eptest_dir_path =  Path("pred_elem_seq/eptest")
    if os.path.isdir(eptest_dir_path):
        shutil.rmtree(eptest_dir_path)
    eptest_dir_path.mkdir()
