import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Any

# Import antares-craft with a clear alias
import antares.craft as ac
from antares.craft import Study


class AntaresStudyUpdater:
    """
    Manages the process of updating an Antares study based on POMMES results.

    This class acts as a container for an Antares study object and provides
    a suite of methods to update its components (areas, links, clusters, etc.)
    using data from a multi-energy system model run.
    """

    def __init__(
        self,
        study: Study,
        model_parameters: xr.Dataset,
        pommes_output: xr.Dataset,
        antares_config: Dict[str, Any],
        load_el: xr.Dataset,
        load_h2: xr.Dataset,
        pecd: xr.Dataset,
        hydro_data: xr.Dataset,
        areas: List[str],
        weather_years: List[int],
        year: int
    ):
        """
        Initializes the updater with the study object and all necessary data.

        Args:
            study: The antares.craft.Study object to be updated.
            model_parameters: The POMMES input parameters dataset.
            pommes_output: The POMMES solution/output dataset.
            antares_config: A dictionary with Antares-specific configuration.
            load_el: xarray Dataset containing electricity load time series.
            load_h2: xarray Dataset containing hydrogen load time series.
            areas: List of area names (e.g., ['FR', 'DE']).
            weather_years: List of weather years for time series data.
        """
        self.study = study
        self.params = model_parameters
        self.output = pommes_output
        self.config = antares_config
        self.load_el = load_el
        self.load_h2 = load_h2
        self.PECD = pecd
        self.hydro_data = hydro_data
        self.areas = areas
        self.weather_years = weather_years
        self.year_op = year  # Assuming single year for now
        print(f"AntaresStudyUpdater initialized for study '{self.study.name}' and year {self.year_op}.")

    def update_areas(self, all_resources: List[str] = ['electricity', 'hydrogen']):
        """
        Creates or updates electricity and hydrogen areas in the study.

        This method sets the unsupplied/spilled energy costs, UI coordinates,
        and load time series for each resource in each specified area.
        """
        print("\n--- Updating Resource Areas ---")
        for resource in all_resources:

            # Define area properties based on the resource
            area_properties = ac.AreaProperties(
                energy_cost_unsupplied=self.params.load_shedding_cost.sel(resource=resource).item(),
                energy_cost_spilled=self.params.spillage_cost.sel(resource=resource).item(),
            )
            load_data = self.load_el if resource == 'electricity' else self.load_h2

            offset = self.config["UI"]["offset"][resource]
            label = self.config['label'][resource]

            for area_name in self.areas:
                area_id = f"{area_name.lower()}_{label}"
                print(f"Processing Area: {area_id}")

                # Configure UI settings
                ui = ac.AreaUi(
                    x=int(self.config["UI"]["country_coordinates"][area_name]['x']),
                    y=int(self.config["UI"]["country_coordinates"][area_name]['y'] + offset),
                    color_rgb=self.config["UI"]["color"][resource],
                )

                # Create or update the area
                if area_id not in self.study.get_areas():
                    self.study.create_area(area_id, properties=area_properties, ui=ui)
                else:
                    self.study.get_areas()[area_id].update_properties(properties=area_properties)
                    self.study.get_areas()[area_id].update_ui(ui=ui)

                area = self.study.get_areas()[area_id]

                # Set the load time series
                df = load_data.sel(node=f"{area_name}00", weather_year=self.weather_years).to_dataframe().reset_index()
                df_pivot = df.pivot(index='time', columns='weather_year', values='demand').fillna(0)
                area.set_load(df_pivot)

    def update_virtual_areas(self):
        """Creates or updates virtual areas used for sector coupling."""
        print("\n--- Updating Virtual Areas ---")
        virtual_areas = self.config.get('virtual_areas', [])
        if not virtual_areas:
            print("No virtual areas defined in config. Skipping.")
            return

        label = self.config['label']['virtual']
        virtual_properties = ac.AreaProperties(energy_cost_unsupplied=0, energy_cost_spilled=0)

        for area_name in virtual_areas:
            area_id = f"{area_name.lower()}_{label}"
            print(f"Processing Virtual Area: {area_id}")

            # Assuming virtual areas use the last resource's offset for positioning
            offset = self.config["UI"]["offset"]["hydrogen"]  # Example positioning

            ui = ac.AreaUi(
                x=int(self.config["UI"]["country_coordinates"][area_name]['x']),
                y=int(self.config["UI"]["country_coordinates"][area_name]['y'] + offset),
                color_rgb=self.config["UI"]["color"]['virtual'],
            )

            if area_id not in self.study.get_areas():
                self.study.create_area(area_id, properties=virtual_properties, ui=ui)
            else:
                self.study.get_areas()[area_id].update_properties(properties=virtual_properties)
                self.study.get_areas()[area_id].update_ui(ui=ui)

    def update_links(self, all_resources: List[str] = ['electricity', 'hydrogen']):
        """
        Creates or updates links between areas for each resource.

        This method identifies unique interconnections from the POMMES output,
        creates corresponding links in the Antares study for each energy vector
        (resource), and sets their transmission capacities.
        """
        print("\n--- Updating Inter-Area Links ---")

        all_pommes_links = self.output.link.values
        study_links = self.study.get_links()

        # 1. Identify unique physical interconnections
        # A link 'FR-DE' is the same as 'DE-FR'. We create a canonical,
        # sorted version to avoid duplicate processing.
        unique_links = set()
        for link_name in all_pommes_links:
            # Check for non-empty string before splitting
            if not link_name or not isinstance(link_name, str):
                continue
            areas_in_link = link_name.split("-")
            # Ensure there are two areas in the link name
            if len(areas_in_link) == 2:
                ordered_link = "-".join(sorted(areas_in_link))
                unique_links.add(ordered_link)

        # 2. Iterate over unique links and resources to update the study
        for link_pommes in unique_links:
            area1_name, area2_name = link_pommes.split("-")

            # POMMES has directional capacities (e.g., FR-DE and DE-FR)
            # We need both to set direct and indirect capacities in Antares.
            link_pommes_direct = f"{area1_name}-{area2_name}"
            link_pommes_indirect = f"{area2_name}-{area1_name}"

            for resource in all_resources:
                label = self.config['label'][resource]
                area_from_id = f"{area1_name.lower()}_{label}"
                area_to_id = f"{area2_name.lower()}_{label}"

                # Antares link names are sorted alphabetically by area name
                link_antares_id = " / ".join(sorted([area_from_id, area_to_id]))

                print(f"Processing Link: {link_antares_id}")

                # 3. Create link if it doesn't exist
                if link_antares_id not in study_links:
                    self.study.create_link(
                        area_from=area_from_id,
                        area_to=area_to_id,
                        properties=ac.LinkProperties()
                    )
                    # Refresh the link list after creation
                    study_links = self.study.get_links()

                link = study_links[link_antares_id]

                # 4. Define and update link properties and UI from config
                link_properties = ac.LinkProperties(
                    hurdles_cost=True,
                    transmission_capacities=ac.TransmissionCapacities.ENABLED,
                    asset_type=self.config['link']['type'][resource],
                    display_comments=True,
                    comments=f"Link for {resource} between {area1_name} and {area2_name}",
                )
                link.update_properties(properties=link_properties)

                ui_config = self.config['link']
                link.update_ui(ui=ac.LinkUi(
                    link_style=ui_config['link_style'][resource],
                    link_width=ui_config['link_width'][resource],
                    colorr=ui_config['color'][resource][0],
                    colorg=ui_config['color'][resource][1],
                    colorb=ui_config['color'][resource][2],
                ))

                # 5. Get capacities from POMMES output and set them in Antares
                try:
                    capa_direct_val = self.output.operation_transport_power_capacity.sel(
                        link=link_pommes_direct,
                        transport_tech=self.config['link']['tech'][resource]
                    ).item()
                except KeyError:
                    capa_direct_val = 0  # Assume 0 if link direction not in output

                try:
                    capa_indirect_val = self.output.operation_transport_power_capacity.sel(
                        link=link_pommes_indirect,
                        transport_tech=self.config['link']['tech'][resource]
                    ).item()
                except KeyError:
                    capa_indirect_val = 0  # Assume 0 if link direction not in output

                # Create 8760-hour long DataFrames for Antares
                df_capa_direct = pd.DataFrame(
                    [int(capa_direct_val)] * 8760, columns=['value']
                )
                df_capa_indirect = pd.DataFrame(
                    [int(capa_indirect_val)] * 8760, columns=['value']
                )

                link.set_capacity_direct(df_capa_direct)
                link.set_capacity_indirect(df_capa_indirect)

    def update_thermal_clusters(
            self,
            conversion_tech_map: Dict[str, List[str]],
            pommes_to_antares_group: Dict[str, str]
    ):
        """
        Creates or updates thermal clusters for each resource in each area.

        This method iterates through specified thermal technologies, calculates
        their marginal costs based on fuel prices, CO2 tax, and efficiencies,
        and then creates or updates the corresponding clusters in Antares with
        the installed capacity from POMMES.

        Args:
            conversion_tech_map: A dictionary mapping a resource (e.g., 'electricity')
                                 to a list of its thermal technologies.
            pommes_to_antares_group: A dictionary mapping POMMES technology names
                                     to Antares cluster group names.
        """
        print("\n--- Updating Thermal Clusters ---")

        carbon_tax = self.params.carbon_tax.item()

        # Define the fuels needed for cost calculation
        fuels = ['methane', 'coal', 'lignite']

        for area_name in self.areas:
            for resource, tech_list in conversion_tech_map.items():
                area_id = f"{area_name.lower()}_{self.config['label'][resource]}"
                area = self.study.get_areas()[area_id]
                area_plants = area.get_thermals()

                for tech in tech_list:
                    # 1. Get capacity from POMMES output
                    capa = self.output.operation_conversion_power_capacity.sel(
                        conversion_tech=tech, area=area_name
                    ).item()
                    capa = int(capa)  # Ensure it's an integer

                    group = pommes_to_antares_group.get(tech, "Other")  # Default to 'Other' if not found

                    print(f"Processing Thermal: {area_name:<5} {resource:<12} {tech:<20} {capa:>6} MW")

                    # 2. Calculate marginal cost
                    emission_factor = self.params.conversion_emission_factor.sel(conversion_tech=tech).item()
                    variable_cost = self.params.conversion_variable_cost.sel(conversion_tech=tech).item()

                    # Calculate fuel cost based on conversion factors and import prices
                    # Note: conversion_factor for fuel is negative in POMMES
                    fuel_factors = self.params.conversion_factor.sel(resource=fuels, conversion_tech=tech, drop=True)
                    fuel_prices = self.params.net_import_import_price.sel(resource=fuels, drop=True)
                    fuel_cost = (-fuel_factors * fuel_prices).sum().item()

                    marginal_cost = round(carbon_tax * emission_factor + variable_cost + fuel_cost, 2)

                    # 3. Define cluster properties
                    # TODO: Implement a heuristic for unit_count and adapt nominal_capacity accordingly
                    thermal_properties = ac.ThermalClusterProperties(
                        group=group,
                        unit_count=1,
                        nominal_capacity=capa,
                        enabled=capa > 0,  # Disable cluster if capacity is zero
                        marginal_cost=marginal_cost,
                        market_bid_cost=marginal_cost,
                        co2=emission_factor,
                    )

                    # 4. Create or update the cluster
                    tech_antares_name = tech.lower()
                    if tech_antares_name in area_plants:
                        thermal_plant = area_plants[tech_antares_name]
                        thermal_plant.update_properties(thermal_properties)
                    elif capa > 0:  # Only create if capacity is non-zero
                        area.create_thermal_cluster(tech_antares_name, properties=thermal_properties)
                        thermal_plant = area.get_thermals()[tech_antares_name]

                    # 5. Set availability series (if the plant exists)
                    if capa > 0:
                        thermal_plant = area.get_thermals()[tech_antares_name]
                        # Set availability to 100% of nominal capacity (as a time series)
                        availability_series = pd.DataFrame([capa] * 8760, columns=['value'])
                        thermal_plant.set_series(availability_series)

    def update_renewable_clusters(
            self,
            vre_tech_list: List[str],
            pommes_to_antares_group: Dict[str, str],
            pommes_to_pecd: Dict[str, str]
    ):
        """
        Creates/updates renewable clusters and their production factor time series.

        Args:
            vre_tech_list: A list of VRE technology names from POMMES.
            pommes_to_antares_group: Maps POMMES tech names to Antares groups.
            pommes_to_pecd: Maps POMMES tech names to PECD dataset tech names.
        """
        print("\n--- Updating Renewable (VRES) Clusters ---")

        for area_name in self.areas:
            # VRES is typically only in the electricity system
            area_id = f"{area_name.lower()}_{self.config['label']['electricity']}"
            try:
                area = self.study.get_areas()[area_id]
                area_renewables = area.get_renewables()
            except KeyError:
                print(f"Warning: Electricity area '{area_id}' not found. Skipping VRES updates for {area_name}.")
                continue

            for tech in vre_tech_list:
                capa = self.output.operation_conversion_power_capacity.sel(
                    conversion_tech=tech, area=area_name
                ).item()
                capa = int(capa)

                print(f"Processing VRES: {area_name:<5} {tech:<20} {capa:>6} MW")

                group = pommes_to_antares_group.get(tech, "other res 1")
                tech_antares_name = tech.lower()

                re_properties = ac.RenewableClusterProperties(
                    group=group,
                    ts_interpretation='production-factor',
                    unit_count=1,
                    nominal_capacity=capa,
                    enabled=capa > 0
                )

                # Create or update the cluster properties
                if tech_antares_name in area_renewables:
                    area_renewables[tech_antares_name].update_properties(re_properties)
                elif capa > 0:
                    area.create_renewable_cluster(tech_antares_name, properties=re_properties)

                # Update time series only if the cluster should be active
                if capa > 0:
                    # Refresh the object in case it was just created
                    renewable_plant = area.get_renewables()[tech_antares_name]
                    pecd_tech_name = pommes_to_pecd.get(tech)

                    if not pecd_tech_name:
                        print(f"Warning: No PECD mapping found for '{tech}'. Skipping series update.")
                        continue

                    # Prepare the production factor time series DataFrame
                    try:
                        df = self.PECD.sel(
                            node=f"{area_name}00",
                            technology=pecd_tech_name,
                            weather_year=self.weather_years
                        ).to_dataframe()

                        df_pivot = df.reset_index().pivot(
                            index='hour', columns='weather_year', values='load_factor'
                        ).fillna(0)

                        renewable_plant.set_series(df_pivot)
                    except KeyError:
                        print(f"Warning: No PECD data for {tech} in {area_name}. Series not updated.")

    def update_st_storage(
            self,
            st_storage_tech_map: Dict[str, List[str]],
            pommes_to_antares_group: Dict[str, str],
            hurdle_cost: float = 0.01
    ):
        """
        Creates or updates short-term storage clusters in each area.

        This method configures properties like injection/withdrawal capacity,
        reservoir capacity, and efficiency for short-term storage technologies.

        Args:
            st_storage_tech_map: Dictionary mapping a resource (e.g., 'electricity')
                                 to a list of its short-term storage technologies.
            pommes_to_antares_group: Maps POMMES tech names to Antares groups.
            hurdle_cost: A small cost to apply to injection/withdrawal to
                         prevent simultaneous operations.
        """
        print("\n--- Updating Short-Term Storage Clusters ---")

        for area_name in self.areas:
            for resource, tech_list in st_storage_tech_map.items():
                area_id = f"{area_name.lower()}_{self.config['label'][resource]}"
                try:
                    area = self.study.get_areas()[area_id]
                    area_storages = area.get_st_storages()
                except KeyError:
                    print(f"Warning: Area '{area_id}' not found. Skipping ST Storage for {area_name}/{resource}.")
                    continue

                for tech in tech_list:
                    # 1. Get capacities from POMMES output
                    capa_power = self.output.operation_storage_power_capacity.sel(
                        storage_tech=tech, area=area_name
                    ).item()
                    capa_energy = self.output.operation_storage_energy_capacity.sel(
                        storage_tech=tech, area=area_name
                    ).item()

                    capa_power = int(capa_power)
                    capa_energy = int(capa_energy)

                    print(
                        f"Processing ST Storage: {area_name:<5} {resource:<12} {tech:<20} P: {capa_power:>6} MW, E: {capa_energy:>6} MWh")

                    # 2. Define ST Storage properties
                    # TODO: Get efficiencies from POMMES parameters when available
                    properties = ac.STStorageProperties(
                        group=pommes_to_antares_group.get(tech, "ST Storage"),
                        injection_nominal_capacity=capa_power,
                        withdrawal_nominal_capacity=capa_power,
                        reservoir_capacity=capa_energy,
                        efficiency=1.0,
                        efficiency_withdrawal=1.0,
                        initial_level=0.5,
                        initial_level_optim=False,
                        enabled=(capa_power > 0 and capa_energy > 0)
                    )

                    # 3. Create or update the storage cluster
                    tech_antares_name = tech.lower()
                    if tech_antares_name in area_storages:
                        storage = area_storages[tech_antares_name]
                        storage.update_properties(properties)
                    elif properties.enabled:  # Only create if it has capacity
                        area.create_st_storage(tech_antares_name, properties=properties)

                    # 4. Set hurdle costs if the storage is active
                    if properties.enabled:
                        storage = area.get_st_storages()[tech_antares_name]
                        cost_df = pd.DataFrame([hurdle_cost] * 8760, columns=['value'])
                        storage.set_cost_injection(cost_df)
                        storage.set_cost_withdrawal(cost_df)

    def _prepare_hydro_inflows_df(self, area_name: str, reservoir_power_capacity: int) -> pd.DataFrame:
        """
        Prepares a DataFrame of daily hydro reservoir inflows from the HYDRO dataset.

        It converts weekly GWh to daily MWh and caps inflows to a reasonable
        daily production capacity of the reservoir.
        """
        try:
            # Convert weekly GWh to daily MWh
            df = self.hydro_data['Reservoir'].sel(node=f"{area_name}00").to_dataframe().reset_index()
            df = df.pivot(index='week', columns='weather_year', values='Reservoir') / 7 * 1e3

            # Expand from weekly to daily format
            df = df.loc[df.index.repeat(7)].reset_index(drop=True)
            df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
            df = df.iloc[:365]  # Ensure exactly 365 days

            # Sanity check: cap daily inflow to avoid unrealistic values
            daily_max_inflow = int(reservoir_power_capacity * 24 * 0.99)
            df = df.map(lambda x: min(int(x), daily_max_inflow) if pd.notna(x) else 0)
            return df

        except KeyError:
            # Return a zero-filled DataFrame if no data exists for the area
            return pd.DataFrame(0, index=pd.RangeIndex(start=1, stop=366), columns=self.weather_years)

    def update_hydro(self, pommes_to_antares_group: Dict[str, str], policy: str = "enabled"):
        """
        Updates hydro reservoir properties and inflows based on a selected policy.

        Args:
            pommes_to_antares_group: Maps POMMES tech names to Antares groups.
            policy (str): The policy to apply for hydro modeling.
                - 'enabled': Standard hydro reservoir with inflows (for Antares >= 9.3).
                - 'disabled': Disables all hydro reservoir functionality.
                - 'workaround': Implements a fix for Antares 9.2 by modeling inflows
                  as a renewable cluster and enabling pumping.
        """
        print(f"\n--- Updating Hydro Reservoirs (Policy: {policy}) ---")
        if policy not in ["enabled", "disabled", "workaround"]:
            raise ValueError(f"Invalid hydro policy: '{policy}'. Must be 'enabled', 'disabled', or 'workaround'.")

        tech = "dam"
        inflow_vre_name = "dam_inflows"  # Name of the temporary VRES cluster
        dam_st_storage_name = "dam"
        for area_name in self.areas:
            area_id = f"{area_name.lower()}_{self.config['label']['electricity']}"
            try:
                area = self.study.get_areas()[area_id]
            except KeyError:
                continue

            # 1. Get base capacities from POMMES
            capa_power = int(self.output.operation_storage_power_capacity.sel(storage_tech=tech,
                                                                              area=area_name).item())
            capa_energy = int(self.output.operation_storage_energy_capacity.sel(storage_tech=tech,
                                                                                area=area_name).item())

            print(f"Processing Hydro: {area_name:<5} P: {capa_power:>6} MW, E: {capa_energy:>6} MWh")

            # 2. Apply the chosen policy
            is_active = capa_power > 0 and capa_energy > 0

            # --- POLICY: ENABLED (Normal Behavior for Antares 9.3+) ---
            if policy == "enabled":
                # Your point #2: Delete the temporary VRES cluster if it exists
                if inflow_vre_name in area.get_renewables():
                    print(f"   - Cleaning up obsolete '{inflow_vre_name}' cluster.")
                    area.delete_renewable_cluster(area.get_renewables()[inflow_vre_name])
                if dam_st_storage_name in area.get_renewables():
                    print(f"   - Cleaning up obsolete '{dam_st_storage_name}' cluster.")
                    area.delete_st_storage(area.get_st_storages()[dam_st_storage_name])

                hydro_props = ac.HydroProperties(reservoir=is_active, reservoir_capacity=capa_energy, follow_load=True,
                                                 use_heuristic=True)
                maxpower_df = pd.DataFrame([[capa_power, 24, 0, 24]] * 365,
                                           columns=["Generating Max Power", "Hours possible at Pmax",
                                                    "Pumping Max Power", "Hours possible at Pmin"])
                inflows_df = self._prepare_hydro_inflows_df(area_name,
                                                            capa_power) if is_active else self._prepare_hydro_inflows_df(
                    area_name, 0)

            # --- POLICY: DISABLED ---
            elif policy == "disabled":
                # Your point #2: Delete the temporary VRES cluster if it exists
                if inflow_vre_name in area.get_renewables():
                    print(f"   - Cleaning up obsolete '{inflow_vre_name}' cluster.")
                    area.delete_renewable_cluster(area.get_renewables()[inflow_vre_name])
                if dam_st_storage_name in area.get_renewables():
                    print(f"   - Cleaning up obsolete '{dam_st_storage_name}' cluster.")
                    area.delete_st_storage(area.get_st_storages()[dam_st_storage_name])

                hydro_props = ac.HydroProperties(reservoir=False, reservoir_capacity=0)
                maxpower_df = pd.DataFrame([[0, 24, 0, 24]] * 365,
                                           columns=["Generating Max Power", "Hours possible at Pmax",
                                                    "Pumping Max Power", "Hours possible at Pmin"])
                inflows_df = self._prepare_hydro_inflows_df(area_name, 0)

            # --- POLICY: WORKAROUND (For Antares 9.2) ---
            elif policy == "workaround":
                # Reservoir is disabled, but pumping is enabled via maxpower
                hydro_props = ac.HydroProperties(reservoir=False, reservoir_capacity=0)
                maxpower_df = pd.DataFrame([[capa_power, 24, capa_power, 24]] * 365,
                                           columns=["Generating Max Power", "Hours possible at Pmax",
                                                    "Pumping Max Power", "Hours possible at Pmin"])
                inflows_df = self._prepare_hydro_inflows_df(area_name, 0)  # Inflows for the hydro unit itself are zero

                # --- Create/Update a VRES cluster to represent inflows ---
                inflow_as_vre_df = self._prepare_hydro_inflows_df(area_name, capa_power)
                hourly_inflows = inflow_as_vre_df.loc[inflow_as_vre_df.index.repeat(24)] / 24
                hourly_inflows.index = pd.RangeIndex(start=0, stop=len(hourly_inflows))
                max_inflow_power = int(hourly_inflows.max().max()) if is_active else 0

                # Your point #1: Use the provided group mapping
                group_name = pommes_to_antares_group.get("dam", "other res 4")

                re_props = ac.RenewableClusterProperties(group=group_name, ts_interpretation='power-generation',
                                                         enabled=is_active, nominal_capacity=max_inflow_power)

                st_stor_props = ac.STStorageProperties(
                        group="other 5",
                        injection_nominal_capacity=capa_power,
                        withdrawal_nominal_capacity=capa_power,
                        reservoir_capacity=capa_energy,
                        efficiency=0.8,#arbitrary

                        efficiency_withdrawal=1,
                        initial_level=0.5,
                        initial_level_optim=False,
                        enabled=is_active
                    )
                if inflow_vre_name not in area.get_renewables():
                    if is_active:  # Only create if needed
                        area.create_renewable_cluster(inflow_vre_name, re_props)
                else:
                    area.get_renewables()[inflow_vre_name].update_properties(re_props)

                if is_active:
                    area.get_renewables()[inflow_vre_name].set_series(hourly_inflows.iloc[:8760])


                if dam_st_storage_name not in area.get_st_storages():
                    if is_active:
                        area.create_st_storage(dam_st_storage_name, properties=st_stor_props)
                else:
                    area.get_st_storages()[dam_st_storage_name].update_properties(st_stor_props)


            # 3. Update Antares study
            area.hydro.update_properties(hydro_props)
            area.hydro.set_maxpower(maxpower_df)
            area.hydro.set_mod_series(inflows_df)

    def update_run_of_river(self):
        """
        Updates the run-of-river generation time series for each area.
        """
        print("\n--- Updating Run-of-River Hydro ---")

        for area_name in self.areas:
            area_id = f"{area_name.lower()}_{self.config['label']['electricity']}"
            try:
                area = self.study.get_areas()[area_id]
            except KeyError:
                continue  # Skip if the electricity area doesn't exist

            # 1. Prepare the RoR inflow time series
            try:
                # Select daily data for the area, convert GWh to MWh, and divide by 24 to get average hourly MW
                df = self.hydro_data['Run_of_River'].sel(node=f"{area_name}00").to_dataframe().reset_index()
                df = df.pivot(index='day', columns='weather_year', values='Run_of_River') / 24 * 1e3

                # Expand from daily to hourly format
                df_hourly = df.loc[df.index.repeat(24)].reset_index(drop=True)
                df_hourly.index = pd.RangeIndex(start=0, stop=len(df_hourly))
                df_hourly = df_hourly.iloc[:8760].map(lambda x: int(x) if pd.notna(x) else 0)

            except KeyError:
                # If no data is found, create a zero-filled DataFrame
                print(f"   - No Run-of-River data found for {area_name}. Setting to zero.")
                df_hourly = pd.DataFrame(0, index=pd.RangeIndex(0, 8760), columns=self.weather_years)

            # 2. Set the time series in the Antares study
            print(f"Processing RoR for: {area_name:<5} Hourly avg: {int(df.mean().mean()):>6} MWh")
            area.hydro.set_ror_series(df_hourly)

    def _get_antares_link_info(self, area_from: str, area_to: str) -> Dict[str, Any]:
        """
        Generates the alphabetically ordered link name and flow direction.
        (Internal utility function)
        """
        sorted_areas = sorted([area_from, area_to])
        flow_direction = 1 if area_from == sorted_areas[0] else -1
        return {
            "link_name": " / ".join(sorted_areas),
            "area_1": sorted_areas[0],
            "area_2": sorted_areas[1],
            "flow_direction": flow_direction,
        }

    def update_sector_coupling(
            self,
            p2g_techs: List[str],
            g2p_techs: List[str],
            pommes_to_antares_group: Dict[str, str],
    ):
        """
        Configures all aspects of sector coupling: P2G/G2P clusters,
        virtual links, and binding constraints.

        This method models Power-to-Gas (P2G) and Gas-to-Power (G2P) conversions
        by creating thermal clusters area corresponding to the final resource
        and constraining their primary resource consumption in the corresponding
        area to the flows going in a virtual sink, enforcing conversion efficiency
        at the same time.

        Args:
            p2g_techs: List of POMMES P2G technology names (e.g., electrolysis).
            g2p_techs: List of POMMES G2P technology names (e.g., ccgt_h2).
            pommes_to_antares_group: A dictionary mapping POMMES technology names
                                     to Antares cluster group names.
        """
        print("\n--- Updating Sector Coupling (Clusters, Links, & Constraints) ---")

        # --- Part 0: Create the P2G and G2P thermal clusters ---
        print("   - Configuring P2G and G2P thermal clusters...")
        p2g_g2p_tech_map = {'hydrogen': p2g_techs, 'electricity': g2p_techs}
        self.update_thermal_clusters(p2g_g2p_tech_map, pommes_to_antares_group)

        # --- Part 1 & 2: Configure Constraints using the helper method ---
        for area_name in self.areas:
            # P2G: electricity -> hydrogen
            self._configure_conversion_constraint(
                area_name=area_name, bc_prefix="p2g",
                input_resource_label=self.config['label']['electricity'],
                output_resource_label=self.config['label']['hydrogen'],
                tech_list=p2g_techs, efficiency_resource='electricity'
            )
            # G2P: hydrogen -> electricity
            self._configure_conversion_constraint(
                area_name=area_name, bc_prefix="g2p",
                input_resource_label=self.config['label']['hydrogen'],
                output_resource_label=self.config['label']['electricity'],
                tech_list=g2p_techs, efficiency_resource='hydrogen'
            )

        # --- Part 3: Update Virtual Sink Load ---
        print("   - Updating virtual sink load...")
        capa_g2p_total = self.output.operation_conversion_power_capacity.sel(conversion_tech=g2p_techs).sum().item()
        capa_p2g_total = self.output.operation_conversion_power_capacity.sel(conversion_tech=p2g_techs).sum().item()
        total_virtual_demand = int(capa_g2p_total + capa_p2g_total)
        sink_id = f"{self.config['virtual_areas'][1].lower()}_{self.config['label']['virtual']}"
        sink_area = self.study.get_areas()[sink_id]
        demand_df = pd.DataFrame([[total_virtual_demand] * len(self.weather_years)] * 8760, columns=self.weather_years)
        sink_area.set_load(demand_df)
        sink_area.update_properties(ac.AreaProperties(
            energy_cost_unsupplied=0, energy_cost_spilled=0,
            non_dispatch_power=False, dispatch_hydro_power=False, other_dispatch_power=False
        ))

    def _configure_conversion_constraint(
                self,
                area_name: str,
                bc_prefix: str,
                input_resource_label: str,
                output_resource_label: str,
                tech_list: List[str],
                efficiency_resource: str,
        ):
            """Helper to configure a virtual link and binding constraint for a conversion technology."""

            # Define area IDs
            input_area_id = f"{area_name.lower()}_{input_resource_label}"
            output_area_id = f"{area_name.lower()}_{output_resource_label}"
            sink_id = f"{self.config['virtual_areas'][1].lower()}_{self.config['label']['virtual']}"

            # Configure virtual link from the input resource to the sink
            link_info = self._get_antares_link_info(input_area_id, sink_id)
            total_capacity = self.output.operation_conversion_power_capacity.sel(
                area=area_name, conversion_tech=tech_list
            ).sum().item()
            self._create_or_update_virtual_link(link_info, int(total_capacity))

            # --- Binding Constraint ---
            bc_name = f"{bc_prefix}-{area_name.lower()}"
            bc = self._create_or_update_binding_constraint(bc_name)

            # Base term for the flow on the virtual link
            terms = [ac.ConstraintTerm(data=ac.LinkData(area1=link_info["area_1"], area2=link_info["area_2"]),
                                       weight=-link_info["flow_direction"])]

            # Add terms for each active technology
            has_active_clusters = False
            for tech in tech_list:
                capa = self.output.operation_conversion_power_capacity.sel(
                    conversion_tech=tech, area=area_name
                ).item()
                if int(capa) > 0:
                    has_active_clusters = True
                    factor = self.params.conversion_factor.sel(conversion_tech=tech,
                                                               resource=efficiency_resource).item()
                    terms.append(ac.ConstraintTerm(data=ac.ClusterData(area=output_area_id, cluster=tech.lower()),
                                                   weight=abs(factor)))

            # Add terms and disable the constraint if it has no active clusters
            if has_active_clusters:
                bc.add_terms(terms)
            else:
                # Disable the constraint if there's no capacity to avoid trivial/empty constraints
                bc.update_properties(ac.BindingConstraintProperties(enabled=False))

        #
        # print("\n--- Updating Sector Coupling Constraints (Clusters, Links, & Constraints) ---")
        # year = self.year_op
        #
        # # --- Part 0: Create the P2G and G2P thermal clusters ---
        # # P2G (electrolysis) PRODUCES hydrogen, so it's a "thermal" plant in the hydrogen area.
        # # G2P (H2 turbines) PRODUCES electricity, so it's a "thermal" plant in the electricity area.
        # print("   - Configuring Power-to-Gas (P2G) and Gas-to-Power thermal clusters...")
        # p2g_g2p_tech_map = {
        #     'hydrogen': p2g_techs,
        #     'electricity': g2p_techs
        # }
        # self.update_thermal_clusters(p2g_g2p_tech_map, pommes_to_antares_group)
        #
        # # --- Part 1: Power-to-Gas (Electrolysis) Links & Constraints ---
        # print("   - Configuring Power-to-Gas (P2G) links and constraints...")
        # for area_name in self.areas:
        #     area_el_id = f"{area_name.lower()}_{self.config['label']['electricity']}"
        #     area_h2_id = f"{area_name.lower()}_{self.config['label']['hydrogen']}"
        #     area_sink_id = f"{self.config['virtual_areas'][1].lower()}_{self.config['label']['virtual']}"
        #
        #     link_info = self._get_antares_link_info(area_el_id, area_sink_id)
        #     capa_tot = self.output.operation_conversion_power_capacity.sel(
        #         area=area_name, year_op=year, conversion_tech=p2g_techs
        #     ).sum().item()
        #
        #     self._create_or_update_virtual_link(link_info, int(capa_tot))
        #
        #     # --- Binding Constraint for P2G ---
        #     bc_name = f"p2g-{area_name.lower()}"
        #     bc = self._create_or_update_binding_constraint(bc_name)
        #
        #     # Equation: (H2_prod * eff_el) - (el_flow_to_sink) = 0
        #     terms = [ac.ConstraintTerm(data=ac.LinkData(area1=link_info["area_1"], area2=link_info["area_2"]),
        #                                weight=-link_info["flow_direction"])]
        #     for tech in p2g_techs:
        #         conv_factor_el = self.params.conversion_factor.sel(conversion_tech=tech, resource='electricity').item()
        #         capa = self.output.operation_conversion_power_capacity.sel(
        #             year_op=year, conversion_tech=tech, area=area_name
        #             ).item()
        #         capa = int(capa)  # Ensure it's an integer
        #         if capa > 0:
        #             terms.append(ac.ConstraintTerm(data=ac.ClusterData(area=area_h2_id, cluster=tech.lower()),
        #                                            weight=abs(conv_factor_el)))
        #     bc.add_terms(terms)
        #
        # # --- Part 2: Gas-to-Power (H2 Combustion) ---
        # print("   - Configuring Gas-to-Power (G2P) links and constraints...")
        # for area_name in self.areas:
        #     area_el_id = f"{area_name.lower()}_{self.config['label']['electricity']}"
        #     area_h2_id = f"{area_name.lower()}_{self.config['label']['hydrogen']}"
        #     area_sink_id = f"{self.config['virtual_areas'][1].lower()}_{self.config['label']['virtual']}"
        #
        #     link_info = self._get_antares_link_info(area_h2_id, area_sink_id)
        #     capa_tot = self.output.operation_conversion_power_capacity.sel(
        #         area=area_name, year_op=year, conversion_tech=g2p_techs
        #     ).sum().item()
        #
        #     self._create_or_update_virtual_link(link_info, int(capa_tot))
        #
        #     # --- Binding Constraint for G2P ---
        #     bc_name = f"g2p-{area_name.lower()}"
        #     bc = self._create_or_update_binding_constraint(bc_name)
        #
        #     # Equation: (el_prod * eff_h2) - (h2_flow_to_sink) = 0
        #     terms = [ac.ConstraintTerm(data=ac.LinkData(area1=link_info["area_1"], area2=link_info["area_2"]),
        #                                weight=-link_info["flow_direction"])]
        #     for tech in g2p_techs:
        #         conv_factor_h2 = self.params.conversion_factor.sel(conversion_tech=tech, resource='hydrogen').item()
        #         capa = self.output.operation_conversion_power_capacity.sel(
        #             year_op=year, conversion_tech=tech, area=area_name
        #             ).item()
        #         capa = int(capa)  # Ensure it's an integer
        #         if capa > 0:
        #             terms.append(ac.ConstraintTerm(data=ac.ClusterData(area=area_el_id, cluster=tech.lower()),
        #                                            weight=abs(conv_factor_h2)))
        #     bc.add_terms(terms)
        #
        # # --- Part 3: Update Virtual Sink Load ---
        # print("   - Updating virtual sink load...")
        # capa_g2p_total = self.output.operation_conversion_power_capacity.sel(year_op=year,
        #                                                                      conversion_tech=g2p_techs).sum().item()
        # capa_p2g_total = self.output.operation_conversion_power_capacity.sel(year_op=year,
        #                                                                      conversion_tech=p2g_techs).sum().item()
        # total_virtual_demand = int(capa_g2p_total + capa_p2g_total)
        #
        # sink_id = f"{self.config['virtual_areas'][1].lower()}_{self.config['label']['virtual']}"
        # sink_area = self.study.get_areas()[sink_id]
        # demand_df = pd.DataFrame([[total_virtual_demand] * len(self.weather_years)] * 8760, columns=self.weather_years)
        # sink_area.set_load(demand_df)
        # sink_area.update_properties(ac.AreaProperties(
        #     energy_cost_unsupplied=0, energy_cost_spilled=0,
        #     non_dispatch_power=False, dispatch_hydro_power=False, other_dispatch_power=False
        # ))

    def _create_or_update_virtual_link(self, link_info: Dict[str, Any], capacity: int):
        """Helper to create and configure a virtual link."""
        study_links = self.study.get_links()
        link_id = link_info["link_name"]

        link_props = ac.LinkProperties(transmission_capacities="enabled", hurdles_cost=False)
        if link_id not in study_links:
            self.study.create_link(area_from=link_info["area_1"], area_to=link_info["area_2"], properties=link_props)
            link = self.study.get_links()[link_id]
        else:
            link = study_links[link_id]
            link.update_properties(link_props)

        capa_df = pd.DataFrame([capacity] * 8760, columns=['value'])
        zero_df = pd.DataFrame([0] * 8760, columns=['value'])
        if link_info["flow_direction"] == 1:
            link.set_capacity_direct(capa_df)
            link.set_capacity_indirect(zero_df)
        else:
            link.set_capacity_direct(zero_df)
            link.set_capacity_indirect(capa_df)

    def _create_or_update_binding_constraint(self, bc_name: str) -> ac.model.binding_constraint.BindingConstraint:
        """Helper to create/update a binding constraint and clear its terms."""
        bc_properties = ac.BindingConstraintProperties(enabled=True, time_step='hourly', operator='equal')

        if bc_name not in self.study.get_binding_constraints():
            self.study.create_binding_constraint(name=bc_name, properties=bc_properties)
        else:
            self.study.get_binding_constraints()[bc_name].update_properties(properties=bc_properties)

        bc = self.study.get_binding_constraints()[bc_name]

        # Robustly delete all existing terms before adding new ones
        existing_terms = list(bc.get_terms().values())
        if existing_terms:
            print(f"   - Clearing {len(existing_terms)} existing term(s) from '{bc_name}'.")
            for term in existing_terms:
                bc.delete_term(term)
        return bc