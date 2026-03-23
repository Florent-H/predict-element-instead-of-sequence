from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats.qmc import LatinHypercube
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
        # write variables data to csv
        var_data.to_csv(
            Path(
                f"pred_elem_seq/datafiles/inputs/variables/weather_invariant/{building.name}_{weather_path.stem}.csv"
            )
        )
    else:
        # write variables data to csv
        var_data.to_csv(
            Path(f"pred_elem_seq/datafiles/inputs/variables/{building.name}.csv")
        )


def get_schedule(schedule_path):
    schedule = pd.read_csv(schedule_path, index_col=0, parse_dates=True)
    return schedule


def write_schedule(building):
    # load EnergyPlus model
    idf_path = Path(f"pred_elem_seq/datafiles/ep/old/{building.name}.idf")
    idf = ef.get_building(idf_path)
    # concatenate objectives and variables
    objs_vars = building.obj_names + building.var_names
    # all parameters are used in the starting points EPProblem
    problem_st_pts = EPProblem(building.parameters, objs_vars)
    # create evaluator with starting points problem and original idf
    evaluator_st_pts = EvaluatorEP(
        problem_st_pts,
        idf,
        out_dir=Path("pred_elem_seq/datafiles"),
        epw=building.weather_path,
    )

    sampler = LatinHypercube(d=len(building.parameters), strength=1)
    factors = sampler.random(1)

    # get starting points dataframe with orthogonal-array-based LHS design
    low_bnd = np.array([param.value_descriptor.min for param in building.parameters])
    up_bnd = np.array([param.value_descriptor.max for param in building.parameters])
    starting_pts = pd.DataFrame(
        data=factors * (up_bnd - low_bnd) + low_bnd,
        columns=[param.name for param in building.parameters],
    )
    # run idf file with any values for the building parameters, just to get variable values for each hour of the
    # year
    res_eval = evaluator_st_pts.df_apply(starting_pts.iloc[0:1], keep_input=True)
    # unravel time domain variables to dataframe
    vars_df = pd.DataFrame(
        data=np.vstack(
            [res_eval[v.name].to_list() for v in building.var_names],
        ).T,
        columns=[v.name for v in building.var_names],
        index=building.sim_time_idx,
        dtype="float64",
    )
    vars_df.to_csv(Path(f"pred_elem_seq/datafiles/schedules/{building.name}.csv"))
