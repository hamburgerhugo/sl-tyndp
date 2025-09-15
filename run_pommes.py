
import pandas as pd
from pommes.io.build_input_dataset import *#build_input_parameters, read_config_file
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
import warnings
import time
from datetime import timedelta

warnings.filterwarnings("ignore")

all_areas=['AL', 'AT', 'BA', 'BE', 'BG', 'CH',
        'CZ', 'DE', 'DK',
         'EE', 'ES', 'FI', 'FR',
         'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU',
         'LV', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT',
        'RO', 'RS', 'SE', 'SI', 'SK', 'UK']

if __name__ == "__main__":
    scenario="GA" #DE or GA
    climate_year=2009 #1995 or 2008 or 2009
    areas = ["FR", "DE"] #, "BE", "ES", "CH", "IT", "PT", "UK", "IE", "AT", "NL", "DK", "NO", "SE", "PL", "CZ"]
    year_op = [2030, 2040, 2050]

    title=f"\033[1m {scenario}CY{climate_year}\033[0m"
    print(title)

    if areas==["all"]:
        areas=all_areas

    add=""
    if len(year_op)<4:
        for i in year_op:
            add+="-"+str(i)
    suffix = f"{len(areas)}-nodes"+add

    print(suffix)
    print(year_op,areas)
    output_folder = f"output/{scenario}_CY{climate_year}/{suffix}"
    start = time.time()
    print("\033[1m Input data load and pre-processing \033[0m")
    config = read_config_file(study=scenario, file_path="config_pommes.yaml")

    ###File pathway update
    if scenario=="DE":
        config["input"]["path"]="data/Distributed Energy"
    elif scenario=="GA":
        config["input"]["path"] = "data/Global Ambition"
    else:
        print(404)
        exit()

    config["coords"]["area"]["values"]=areas
    config["coords"]["year_op"]["values"] = year_op

    # config["input"]["parameters"]["demand"]["file"]=f"demand_cy{climate_year}.csv"
    config["input"]["parameters"]["conversion_availability"]["file"] = f"availability_cy{climate_year}.csv"
    config["input"]["parameters"]["conversion_max_yearly_production"]["file"] = f"conversion_op2_cy{climate_year}.csv"
    config["input"]["parameters"]["conversion_power_capacity_max"]["file"] = f"conversion_op2_cy{climate_year}.csv"
    config["input"]["parameters"]["conversion_power_capacity_min"]["file"] = f"conversion_op2_cy{climate_year}.csv"

    ####Links adjustment
    if config["add_modules"]["transport"] :
        areas=config["coords"]["area"]["values"]
        all_links=pd.read_csv(config["input"]["path"]+"/transport_link.csv",sep=";").link.unique()
        links = []

        for link in all_links:
            pos = ""
            i = 0
            while pos != "-":
                pos = link[i]
                i += 1
            area_from = link[:i - 1]
            area_to = link[i:]
            if area_to in areas and area_from in areas:
                links.append(link)
        if len(links) >= 1:
            config["coords"]["link"]["values"] = links
        else:
            config["add_modules"]["transport"]=False
    print("Transport activated:", config["add_modules"]["transport"])
    model_parameters = build_input_parameters(config)
    model_parameters = check_inputs(model_parameters)
    print("\033[1m Model building \033[0m")
    model = build_model(model_parameters)



    from pathlib import Path

    # Get existing coordinates
    resource = ['electricity', 'hydrogen']
    selected_weather_year = 2009
    selected_nodes = model_parameters.coords['area'].values
    # Define the path
    windows_path = r"C:\Users\hamburgerhug\Documents\These\Data\tyndp-sl-data\workflow_outputs"

    # Initialize an empty list to collect datasets
    dataset_list = []
    for year in year_op:
        # update demand
        for res in resource:
            if res == 'hydrogen':
                file_path = Path(windows_path) / f'demand/NT/H2/NT_H2_{year}_uniform.nc'
            else:
                file_path = Path(windows_path) / f'demand/NT/{res.capitalize()}/NT_{res.capitalize()}_{year}.nc'
            ds = xr.open_dataset(file_path)
            if res == 'hydrogen':
                ds=ds.drop_vars('scenario').squeeze('scenario')
            # Expand dimensions
            ds = ds.expand_dims({'resource': [res], 'year_op': [year]})
            # Assuming `ds` is your dataset and 'node' is a coordinate
            nodes = ds.node.values
            # Step 1: Filter nodes ending with '00'
            filtered_nodes = [n for n in nodes if n.endswith('00')]
            # Step 2: Remove '00' from those node names
            renamed_nodes = [n[:-2] for n in filtered_nodes]
            # Step 3: Create a new DataArray with updated node names
            # Select only the relevant data
            ds = ds.sel(node=filtered_nodes)
            # Update the coordinate
            ds = ds.assign_coords(node=("node", renamed_nodes))
            # Filter and average over weather_year if needed
            valid_nodes = [n for n in selected_nodes if n in sorted(ds.node.values)]
            ds = ds['demand'].sel(weather_year=selected_weather_year, node=valid_nodes)
            # Step 1: Replace datetime64 time coordinate with hour index
            n_hours = ds.sizes['time']
            ds = ds.assign_coords(time=np.arange(1, n_hours + 1).astype('int64'))
            # Step 2: Rename the coordinate from 'time' to 'hour'
            ds = ds.rename({'time': 'hour', 'node': 'area'})
            # Expand dimensions
            ds = ds.drop_vars('weather_year')
            ds = ds.transpose('area', 'hour', 'resource', 'year_op')

            # Append to list
            dataset_list.append(ds)

    # Merge all datasets into one
    merged_ds = xr.merge(dataset_list)
    merged_ds = merged_ds.fillna(0)

    # Align merged_ds to the shape of model_parameters.demand
    model_parameters['demand'].loc[
        dict(
            area=merged_ds.area,
            hour=merged_ds.hour,
            resource=merged_ds.resource,
            year_op=merged_ds.year_op
        )
    ] = merged_ds['demand']
    model_parameters = check_inputs(model_parameters)

    ################################################################################
    #   update conversion_availability
    ################################################################################

    file_path = Path(windows_path) / f'availability/pecd_2040.nc'
    ds = xr.open_dataset(file_path)

    file_path = Path(windows_path) / f'availability/hydro_dataset.nc'
    ds = xr.open_dataset(file_path)
    #TODO hour clustering ... temporal reduction


    print("\033[1m Model solving \033[0m")
    model.solve(solver_name="gurobi", threads=0, method=2, crossover=0,
                logtoconsole=1,outputflag=1,presparsify=2,#barhomogeneous=1,
                nodefilestart=0.1)
    converge = True
    print(model.termination_condition)
    if model.termination_condition not in ["optimal","suboptimal"]:
        try:
            print("\t Searching for infeasabilities")
            model.compute_infeasibilities()
            model.print_infeasibilities()
            converge = False
        except:
            pass
    if converge:
        print("\033[1m Results export \033[0m")
        save_solution(
            model=model,
            output_folder=output_folder,
            model_parameters=model_parameters,
        )

        elapsed_time = time.time() - start
        print("Process took {}".format(timedelta(seconds=elapsed_time)))
