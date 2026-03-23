from pathlib import Path
import pandas as pd
import shutil


def write_var_data(building, variables, weather_path=None):
    # get IDF with Output:Variables
    idf_with_vars = Path(
        f"pred_elem_seq/datafiles/ep/with variables/{building.name}_with_vars.idf"
    )
    # assign weather path to building object if provided
    if isinstance(weather_path, Path):
        building.weather_path = weather_path
    # create eprun_dir
    building.make_eprun_dir()
    # copy idf to eprun directory
    shutil.copy(idf_with_vars, building.eprun_dir / idf_with_vars.name)
    # run EnergyPlus
    building.run(idf_with_vars)
    # get simulation data by reading csv output file from Energy Plus
    output_path = building.eprun_dir / f"{building.name}_with_varsout.csv"
    # gather weather and schedule variables
    output_csv = pd.read_csv(output_path, index_col=0)
    var_data = output_csv.iloc[:, : -building.n_objs]
    # set index of variables dataframe to sim_time_idx
    var_data.index = building.sim_time_idx
    # include hour of day and day of week variables
    var_data.loc[:, "hourOfDay"] = building.sim_time_idx.hour
    var_data.loc[:, "dayOfWeek"] = building.sim_time_idx.dayofweek
    # rename columns to variable names
    var_data.columns = variables
    # set fHeat and fCool to be 1 or 0 depending on if the day or night setpoint temperature is used
    var_data.loc[:, "fTHeat"] = var_data.loc[:, "fTHeat"] // max(
        var_data.loc[:, "fTHeat"]
    )
    var_data.loc[:, "fTCool"] = 1 - var_data.loc[:, "fTCool"] // max(
        var_data.loc[:, "fTCool"]
    )

    if isinstance(weather_path, Path):
        weather_var_path = Path("pred_elem_seq/datafiles/inputs/variables/weather_invariant")
        weather_var_path.mkdir(parents=True, exist_ok=True)
        # write variables data to csv
        var_data.to_csv(
            weather_var_path / f"{building.name}_{weather_path.stem}.csv"
            )
    else:
        weather_var_path = Path("pred_elem_seq/datafiles/inputs/variables")
        weather_var_path.mkdir(parents=True, exist_ok=True)
        # write variables data to csv
        var_data.to_csv(
            weather_var_path / f"{building.name}.csv"
        )
