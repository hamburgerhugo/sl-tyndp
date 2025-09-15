import pandas as pd
from typing import Dict, Any

import antares.craft as ac
from antares.craft.model.output import Output
from antares.craft.model.simulation import JobStatus
from antares.craft.service.api_services.factory import read_study_api

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


def get_outputs_from_runner(runner: 'AntaresRunner' ):
    """
    Fetches the Output object for every successful job in a completed AntaresRunner.
    This function is based on the actual data structure of runner.jobs.

    Args:
        runner: A completed AntaresRunner object containing job and study objects.

    Returns:
        A dictionary mapping each successful year to its Output object.
    """
    print("\n--- Fetching All Simulation Outputs from API ---")
    outputs_by_year = {}


        # Correctly iterate through the runner.jobs dictionary
    for year, info in runner.jobs.items():
        # Correctly access the 'job' and 'study' keys
        job = info.get("job")
        study_id = info.get("study").service.study_id

        #reload study to get outputs
        study = read_study_api(api_config=runner.api_config, study_id=study_id)

        # Check if the necessary objects and statuses are present and correct
        if not (job and study and job.status == JobStatus.SUCCESS and job.output_id):
            if job:
                print(f"Skipping year {year} due to job status: {job.status.value}")
            continue

        try:
            print(f"Fetching output for year {year} (Study: '{study.name}', Output ID: {job.output_id})...")

            # Get the output object from the study
            output_obj = study.get_output(job.output_id)
            outputs_by_year[year] = output_obj

        except Exception as e:
            print(f"!! ERROR: Could not fetch output for year {year}. Reason: {e}")

    print(f"Successfully fetched {len(outputs_by_year)} output(s).")
    return outputs_by_year


class AntaresResultsReader:
    """
    A wrapper class for a single Antares Output object to provide easy access
    to processed results and plots.
    """

    def __init__(self, output: Output):
        """
        Initializes with a single antares.craft.model.output.Output object.
        """
        if not isinstance(output, Output):
            raise TypeError("AntaresResults must be initialized with an Antares Output object.")
        self.output = output

    def get_adequacy_metrics(self) -> pd.DataFrame:
        """
        Extracts key adequacy metrics (LOLE, Unserved Energy) for all areas.
        """
        try:
            df = self.output.aggregate_mc_all_areas(data_type="values", frequency="annual")

            required_cols = ["UNSP. ENRG", "LOLD"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame()

            adequacy_df = df.loc[:, required_cols].copy()
            adequacy_df = adequacy_df.rename(columns={
                "UNSP. ENRG": "unserved_energy_mwh",
                "LOLD": "lole_h"
            })
            return adequacy_df
        except Exception as e:
            print(f"Could not read adequacy metrics: {e}")
            return pd.DataFrame()

    def plot_weekly_balance(self, area_id: str, week_number: int, mc_year: int = 1):
        """
        Generates and displays a stacked area plot of the energy balance for a specific week.

        Args:
            area_id: The full ID of the area (e.g., 'fr_el').
            week_number: The week to plot (1-52).
            mc_year: The Monte Carlo year to extract (1-based index).
        """
        if not _PLOT_AVAILABLE:
            print("Plotting libraries not found. Please install matplotlib and seaborn.")
            return

        try:
            # Get hourly data for the entire year
            hourly_df_all = self.output.get_mc_years_data(
                area_id=area_id, data_type="details", frequency="hourly"
            )[str(mc_year)]

            # Select the specific week
            start_hour = (week_number - 1) * 168
            end_hour = week_number * 168
            hourly_df = hourly_df_all.iloc[start_hour:end_hour].copy()

            if hourly_df.empty:
                print(f"No data found for week {week_number}.")
                return

            # Define generation types and their colors
            gen_types = {
                'NUCLEAR': 'darkred', 'LIGNITE': 'saddlebrown', 'COAL': 'black',
                'GAS': 'gray', 'OIL': 'purple', 'MIX. FUEL': 'olive', 'MISC. DTG': 'pink',
                'H. ROR': 'aqua', 'WIND': 'skyblue', 'SOLAR': 'gold',
                'H. STOR': 'blue', 'PSP': 'darkblue', 'BATTERY': 'limegreen'
            }
            gen_cols = [col for col in gen_types if col in hourly_df.columns]

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(15, 7))

            # Plot stacked generation
            ax.stackplot(hourly_df.index, hourly_df[gen_cols].T, labels=gen_cols,
                         colors=[gen_types[c] for c in gen_cols])

            # Plot load and net load
            ax.plot(hourly_df.index, hourly_df['LOAD'], color='red', linewidth=2.5, label='Load')

            ax.set_title(f"Energy Balance for '{area_id}' - Week {week_number} (MC Year {mc_year})")
            ax.set_xlabel("Hour of the Week")
            ax.set_ylabel("Power (MW)")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)
            ax.margins(x=0, y=0)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not generate plot for '{area_id}': {e}")