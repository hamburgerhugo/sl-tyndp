# sl_tyndp/antares_io.py

import os
from urllib.parse import quote
# Import antares-craft with a clear alias
import antares.craft as ac
import numpy as np
from antares.craft import APIconf, Study
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

import xarray as xr
from pommes.io.build_input_dataset import build_input_parameters, read_config_file
from pommes.model.data_validation.dataset_check import check_inputs


def load_api_config_from_yaml(config_path: Path) -> APIconf:
    """
    Loads Antares API configuration from a YAML file and sets up proxy.

    This function reads API host, token, and proxy details from a YAML file,
    configures the necessary environment variables for the proxy, and returns
    an initialized APIconf object.

    Args:
        config_path: Path to the local configuration YAML file.

    Returns:
        An initialized antares.craft.APIconf object.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        KeyError: If essential keys are missing from the config file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract required values, raising KeyError if they are missing
    try:
        api_host = config["antares_api_host"]["values"]
        token = config["antares_token"]["values"]
        proxy_id = config["proxy_id"]["values"]
        proxy_pwd = quote(config["proxy_pwd"]["values"])  # URL-encode password
        proxy_host = config["proxy_host"]["values"]
        proxy_port = config["proxy_port"]["values"]
    except KeyError as e:
        raise KeyError(f"Missing essential key in {config_path}: {e}")

    # Set up proxy environment variables
    proxy_url = f"http://{proxy_id}:{proxy_pwd}@{proxy_host}:{proxy_port}"
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['NO_PROXY'] = "localhost,127.0.0.1,.rte-france.com"

    print(f"Proxy configured for {proxy_host}:{proxy_port}")

    return APIconf(api_host=api_host, token=token, verify=False)


def get_or_create_study(
        api_config: APIconf,
        study_id: Optional[str],
        study_name: str,
        antares_version: str = "9.2"
) -> Study:
    """
    Retrieves an existing Antares study from the API or creates a new one.

    If a study_id is provided, it attempts to load the study. If loading fails
    or if no study_id is provided, it creates a new study with the given name
    and version.

    Args:
        api_config: The API configuration object.
        study_id: The ID of an existing study to load. Can be None.
        study_name: The name for the study if it needs to be created.
        antares_version: The Antares version for a new study.

    Returns:
        The loaded or newly created antares.craft.Study object.
    """
    study = None
    if study_id:
        try:
            study = ac.read_study_api(api_config=api_config, study_id=study_id)
            print(f"Successfully loaded Antares study '{study.name}' (ID: {study.service.study_id}) from API.")
        except Exception as e:
            print(f"Could not load study with ID '{study_id}'. A new one will be created. Reason: {e}")
    if not study:
        try:
            study = ac.create_study_api(
                study_name=study_name,
                version=antares_version,
                api_config=api_config
            )
            print(f"Successfully created new Antares study '{study.name}' (ID: {study.service.study_id}) on API.")
        except Exception as e:
            print(f"Fatal error: Could not create new study '{study_name}'.")
            raise e  # Re-raise the exception to halt execution

    return study


def update_study_id_in_config(config_path: Path, new_study_id: str):
    """
    Updates the study ID in the local YAML configuration file.

    This is useful for persisting the ID of a newly created study across sessions.

    Args:
        config_path: Path to the local configuration YAML file.
        new_study_id: The new study ID to save.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the study ID
    if "antares_study" not in config:
        config["antares_study"] = {}
    config["antares_study"]["id"] = new_study_id

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Updated study ID in '{config_path}' to '{new_study_id}'.")


def update_variant_ids_in_config(config_path: Path, variant_ids: Dict[int, str]):
    """
    Updates the year-specific study (variant) IDs in the run configuration file.

    This function reads the specified YAML file, updates the dictionary under the
    'study_ids_by_year' key, and writes the file back.

    Args:
        config_path: Path to the run configuration YAML file (e.g., config_run.yaml).
        variant_ids: A dictionary mapping years to their study IDs to be saved.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure the top-level key exists before trying to update it
    if 'study_ids_by_year' not in config or not isinstance(config['study_ids_by_year'], dict):
        config['study_ids_by_year'] = {}

    # Update the dictionary with the new or confirmed IDs
    config['study_ids_by_year'].update(variant_ids)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated variant IDs in '{config_path}': {variant_ids}")


def load_run_config(config_path: Path) -> Dict[str, Any]:
    """Loads the main run configuration YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Run configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _preprocess_hydro_data(raw_hydro_ds: xr.Dataset, areas: List[str], year: int,
                           weather_years: List[int]) -> xr.Dataset:
    """Aggregates and cleans the raw HYDRO dataset."""

    hydro = raw_hydro_ds.sel(weather_year=weather_years, year=year)

    NATIONAL_AGGREGATION_CONFIG = {
        'SE00': ['SE01', 'SE02', 'SE03', 'SE04'],
        'NO00': ['NOM1', 'NON1', 'NOS0'],
        'IT00': ['ITCA', 'ITCN', 'ITN1', 'ITS1', 'ITSI', 'ITSA'],
    }

    # Aggregate national nodes
    national_agg = xr.concat([
        hydro.sel(node=subnodes).sum(dim='node').expand_dims(node=[key])
        for key, subnodes in NATIONAL_AGGREGATION_CONFIG.items()
    ], dim='node')

    # Select other area nodes
    selected_nodes = [area + '00' for area in areas if area + '00' in hydro.node.values]
    area_selection = hydro.sel(node=selected_nodes)
    merged = xr.concat([area_selection, national_agg], dim='node')

    # Fill missing nodes with zeros
    all_expected_nodes = [area + '00' for area in areas]
    existing_nodes = merged.node.values.tolist()
    missing_nodes = [node for node in all_expected_nodes if node not in existing_nodes]

    if missing_nodes:
        template = merged.isel(node=0).drop_vars('node')
        zero_data = xr.zeros_like(template).expand_dims(node=missing_nodes)
        merged = xr.concat([merged, zero_data], dim='node')

    return merged.fillna(0)


def _fill_missing_nodes(ds: xr.Dataset, all_target_nodes: List[str]) -> xr.Dataset:
    """Ensures a dataset contains all target nodes, filling missing ones with zeros."""
    existing_nodes = ds.node.values.tolist()
    missing_nodes = [node for node in all_target_nodes if node not in existing_nodes]

    if not missing_nodes:
        return ds  # No changes needed

    print(f"   - INFO: Adding zero-filled data for missing nodes: {missing_nodes}")

    # Create a template of zero data based on the first existing node
    template = ds.isel(node=0).drop_vars('node')
    zero_data_for_vars = {var: (template[var].dims, np.zeros_like(template[var].values)) for var in ds.data_vars}
    zero_ds_template = xr.Dataset(zero_data_for_vars, coords=template.coords)

    # Expand this zero template for all missing nodes
    zero_ds = zero_ds_template.expand_dims(node=missing_nodes)

    # Combine the original and the new zero-filled dataset
    return xr.concat([ds, zero_ds], dim='node').sortby("node")


def load_input_datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads all necessary xarray datasets based on the run configuration.
    This version handles both shared and year-specific data files and
    robustly pads any missing nodes with zero-filled data.

    Args:
        config: The loaded run configuration dictionary.

    Returns:
        A dictionary containing the loaded xarray datasets. 'raw_hydro_data'
        is returned without year selection, while others are merged into
        multi-year datasets.
    """
    print("\n--- Loading All Input Datasets ---")
    paths = config['file_paths']
    years_to_run = config['study_parameters']['years_to_run']
    areas = config['study_parameters']['areas']
    weather_years = list(range(
        config['study_parameters']['weather_years']['start'],
        config['study_parameters']['weather_years']['end'] + 1
    ))
    target_nodes = [area + '00' for area in areas]

    # --- 1. Load SHARED datasets (used for all years) ---
    print("Loading shared datasets...")
    shared_paths = paths['shared']
    output_ds = xr.open_dataset(shared_paths['pommes_output'])
    raw_hydro_ds = xr.open_dataset(shared_paths['raw_hydro_data'])

    # Load and process the shared PECD file
    pecd_shared_ds = xr.open_dataset(shared_paths['pecd'])
    available_pecd_nodes = [node for node in target_nodes if node in pecd_shared_ds.node.values]
    pecd_filtered = pecd_shared_ds.sel(node=available_pecd_nodes, weather_year=weather_years)
    pecd_ds = _fill_missing_nodes(pecd_filtered, target_nodes)

    # --- 2. Load YEAR-SPECIFIC datasets ---
    print("Loading year-specific datasets...")
    by_year_paths = paths['by_year']

    def _load_and_combine_years(path_template: str, profile: str = "") -> xr.Dataset:
        """Helper to load year-specific files, pad them, and combine."""
        ds_list = []
        for year in years_to_run:
            try:
                path = Path(path_template.format(year=year, profile=profile))
                ds = xr.open_dataset(path)

                available_nodes = [node for node in target_nodes if node in ds.node.values]
                ds_filtered = ds.sel(node=available_nodes, weather_year=weather_years)
                ds_padded = _fill_missing_nodes(ds_filtered, target_nodes)

                if 'year_op' not in ds_padded.coords:
                    ds_padded = ds_padded.expand_dims(year_op=[year])
                ds_list.append(ds_padded)
            except FileNotFoundError:
                print(f"Warning: Data file not found for year {year} at {path}. A zero-filled dataset will be used.")
                # Create a zero-filled dataset as a fallback
                template = pecd_ds  # Use a known good dataset as a template for coords
                zero_data = xr.zeros_like(
                    template.isel(time=0, drop=True).rename({"demand": "demand"}).expand_dims(time=8760))
                zero_data = zero_data.expand_dims(year_op=[year])
                ds_list.append(zero_data)

        return xr.concat(ds_list, dim='year_op')

    h2_profile = "_uniform" if config['model_settings']['uniform_h2_profile'] else ""
    load_el_ds = _load_and_combine_years(by_year_paths['load_el'])
    load_h2_ds = _load_and_combine_years(by_year_paths['load_h2'], profile=h2_profile)

    print("All datasets loaded successfully.")

    return {
        "pommes_output": output_ds,
        "pecd": pecd_ds,
        "load_el": load_el_ds,
        "load_h2": load_h2_ds,
        "raw_hydro_data": raw_hydro_ds
    }

def get_or_create_variant(
    base_study: Study,
    variant_name: str,
    variant_id: Optional[str] = None
) -> Study:
    """
    Retrieves an existing study variant by its ID or creates a new one.

    This function is crucial for iterative workflows, allowing reuse of
    study variants across multiple runs.

    Args:
        base_study: The parent study from which to create the variant.
        variant_name: The desired name of the variant. This will be used if a new
                      variant needs to be created.
        variant_id: The ID of an existing variant study. If provided and valid,
                    this study will be loaded.

    Returns:
        The loaded or newly created antares.craft.Study object for the variant.
    """
    if variant_id:
        try:
            # Attempt to load the variant directly using its known ID
            variant_study = read_study_api(api_config=base_study.service.config, study_id=variant_id)
            print(f"Successfully loaded existing variant '{variant_study.name}' (ID: {variant_id}).")
            return variant_study
        except Exception:
            print(f"Could not load variant with ID '{variant_id}'. A new one will be created.")

    # If loading failed or no ID was provided, create a new variant
    print(f"Creating new variant named '{variant_name}'...")
    variant_study = base_study.create_variant(variant_name)
    print(f"Successfully created new variant '{variant_study.name}' (ID: {variant_study.id}).")
    return variant_study


def prepare_data_for_batch(run_config: Dict[str, Any]) -> Dict[int, Any]:
    """
    Main function to load all data and prepare the year-specific data batch.

    This function orchestrates the loading of:
    1. POMMES model parameters.
    2. All necessary ANTARES time series datasets.

    It then processes and slices this data to create a dictionary formatted
    for the AntaresRunner.

    Args:
        run_config: The main run configuration dictionary.

    Returns:
        A dictionary where keys are simulation years and values are data
        dictionaries ready for the AntaresStudyUpdater.
    """

    # --- 1. Load POMMES Model Parameters ---
    print("\n--- Loading POMMES Model Parameters ---")
    pommes_settings = run_config['pommes_settings']

    pommes_config = read_config_file(study=pommes_settings['scenario'], file_path=pommes_settings['config_path'])

    # Update pommes_config with values from run_config
    pommes_config["coords"]["area"]["values"] = run_config['study_parameters']['areas']
    pommes_config["input"]["path"] = pommes_settings['base_data_path']

    # Update year-specific file names
    cy = pommes_settings['weather_year']
    pommes_config["input"]["parameters"]["conversion_max_yearly_production"]["file"] = f"conversion_op2_cy{cy}.csv"
    pommes_config["input"]["parameters"]["conversion_power_capacity_max"]["file"] = f"conversion_op2_cy{cy}.csv"
    pommes_config["input"]["parameters"]["conversion_power_capacity_min"]["file"] = f"conversion_op2_cy{cy}.csv"

    # Adjust transport links based on selected areas
    _adjust_transport_links(pommes_config)

    model_parameters = build_input_parameters(pommes_config)
    model_parameters = check_inputs(model_parameters)
    print("POMMES parameters loaded successfully.")

    # --- 2. Load ANTARES Datasets ---
    datasets = load_input_datasets(run_config)

    # --- 3. Prepare the Final Batch Dictionary ---
    print("\n--- Preparing Data Batch for Runner ---")
    year_data_batch = {}
    years_to_run = run_config['study_parameters']['years_to_run']
    areas = run_config['study_parameters']['areas']
    weather_years = list(range(
        run_config['study_parameters']['weather_years']['start'],
        run_config['study_parameters']['weather_years']['end'] + 1
    ))
    study_ids_from_config = run_config.get('study_ids_by_year', {})
    tech_maps = run_config['technology_mappings']
    settings = run_config['model_settings']

    for year in years_to_run:
        hydro_data_for_year = _preprocess_hydro_data(
            datasets["raw_hydro_data"], areas, year, weather_years
        )
        # ).squeeze("year", drop=True))  .drop_vars("year")
        year_data_batch[year] = {
            "study_id": study_ids_from_config.get(year),
            "year": year,
            "model_parameters": model_parameters.sel(year_op=year).drop_vars("year_op"),
            "pommes_output": datasets["pommes_output"].sel(year_op=year).drop_vars("year_op"),
            "antares_config": run_config['antares_config'],  # Pass the antares-specific config
            "load_el": datasets["load_el"].sel(year_op=year).drop_vars("year_op"),
            "load_h2": datasets["load_h2"].sel(year_op=year).drop_vars("year_op"),
            "pecd": datasets["pecd"],
            "hydro_data": hydro_data_for_year,
            "areas": areas,
            "weather_years": weather_years,
            'thermal_tech_map': {'electricity': tech_maps['thermal_el'], 'hydrogen': tech_maps['thermal_h2']},
            'st_storage_map': {'electricity': tech_maps['st_storage_el'], 'hydrogen': tech_maps['st_storage_h2']},
            'vre_techs': tech_maps['vre'],
            'p2g_techs': tech_maps['p2g'],
            'g2p_techs': tech_maps['g2p'],
            'pommes_to_antares_group': tech_maps['pommes_to_antares_group'],
            'pommes_to_pecd': tech_maps['pommes_to_pecd'],
            'hydro_policy': settings['hydro_policy'],
            'hurdle_cost': settings['hurdle_cost']
        }
    print("Data batch preparation complete.")
    return year_data_batch


def _adjust_transport_links(pommes_config: Dict[str, Any]):
    """Helper to filter transport links based on the selected areas."""
    if pommes_config["add_modules"]["transport"]:
        areas = pommes_config["coords"]["area"]["values"]
        all_links_path = Path(pommes_config["input"]["path"]) / "transport_link.csv"
        all_links = pd.read_csv(all_links_path, sep=";").link.unique()

        valid_links = []
        for link in all_links:
            area_from, area_to = link.split("-")
            if area_from in areas and area_to in areas:
                valid_links.append(link)

        if valid_links:
            pommes_config["coords"]["link"]["values"] = valid_links
            print(f"Transport activated with {len(valid_links)} links.")
        else:
            pommes_config["add_modules"]["transport"] = False
            print("Transport disabled: no valid links found for selected areas.")