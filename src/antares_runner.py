# src/antares_runner.py

import time
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

import antares.craft as ac
from antares.craft import Study
from src.antares_io import get_or_create_variant
from src.antares_updater import AntaresStudyUpdater
from src.antares_reader import AntaresResultsReader
from antares.craft.service.api_services.factory import read_study_api


class AntaresRunner:
    """
    Manages creating, updating, running, and collecting results for multiple ANTARES studies.
    """

    def __init__(self, api_config: ac.APIconf, study_version: str, solver_name: str = "xpress", nb_cpu: int = 8):
        """
        Args:
            api_config: The API configuration object.
            study_version: The Antares version for creating new studies (e.g., "9.2").
            solver_name: The solver to use ('xpress' or 'sirius').
            nb_cpu: Number of CPUs for the simulation.
            jobs: Stores the job object AND the full study object for each year
        """
        self.api_config = api_config
        self.study_version = study_version
        self.solver_name = solver_name
        self.nb_cpu = nb_cpu
        self.jobs: Dict[str, Any] = {}

    def run_batch(self, year_data: Dict[int, Dict[str, Any]], study_name_prefix: str):
        """
        Gets/creates, updates, and launches a classical study for each year.

        Args:
            year_data: A dictionary where keys are simulation years and values are
                       data dictionaries. Each dict can contain a 'study_id'.
            study_name_prefix: The base name for creating new studies (e.g., "HH-SL-0.1.1").

        Returns:
            A dictionary mapping each year to its definitive study ID.
        """
        print(f"\n--- Starting Full Batch Run for Years: {list(year_data.keys())} ---")

        definitive_study_ids = {}

        for year, data in year_data.items():
            year_study = None
            study_id = data.get("study_id")
            study_name = f"{study_name_prefix}_run_{year}"

            #TODO refactor this part to call the method individually
            # 1. Get or Create the year-specific study
            if study_id:
                try:
                    year_study = read_study_api(api_config=self.api_config, study_id=study_id)
                    print(f"Successfully loaded study for year {year} (ID: {study_id}).")
                except Exception:
                    print(f"Could not load study with ID '{study_id}'. Creating a new study.")

            if not year_study:
                print(f"Creating new classical study '{study_name}' for year {year}...")
                year_study = ac.create_study_api(
                    study_name=study_name,
                    version=self.study_version,
                    api_config=self.api_config
                )
                print(f"Successfully created new study with ID: {year_study.service.study_id}")

            definitive_study_ids[year] = year_study.service.study_id

            # TODO refactor this part to call the method individually
            try:
                # 2. Prepare the data for the updater's constructor
                # THE FIX IS HERE: We only pass the arguments that __init__ expects.
                init_keys = [
                    "model_parameters", "pommes_output", "antares_config",
                    "load_el", "load_h2", "pecd", "hydro_data",
                    "areas", "weather_years", "year"
                ]
                updater_init_data = {key: data[key] for key in init_keys}

                # 3. Initialize the Updater with the correct data
                updater = AntaresStudyUpdater(study=year_study, **updater_init_data)

                # 4. Execute the full update workflow using the full 'data' dictionary
                print(f"Executing full update for study {year_study.name}...")
                updater.update_areas() # First Iteration only
                updater.update_virtual_areas() # First Iteration only
                updater.update_links()
                updater.update_thermal_clusters(data['thermal_tech_map'], data['pommes_to_antares_group'])
                updater.update_renewable_clusters(data['vre_techs'], data['pommes_to_antares_group'], data['pommes_to_pecd']) # First Iteration only for the load factor
                updater.update_st_storage(data['st_storage_map'], data['pommes_to_antares_group'], data['hurdle_cost'])
                updater.update_hydro(data['pommes_to_antares_group'], data['hydro_policy'])  # check if hydro is constant but can be First Iteration only
                updater.update_run_of_river()  # check if hydro is constant but can be  First Iteration only
                updater.update_sector_coupling(data['p2g_techs'], data['g2p_techs'], data['pommes_to_antares_group']) # First Iteration only for the binding constraints ?
                print(f"Update complete for study {year_study.name}.")

                # TODO refactor this part to call the method individually
                # 5. Launch the simulation
                params = ac.AntaresSimulationParameters(
                    solver=ac.Solver.XPRESS if self.solver_name == "xpress" else ac.Solver.SIRIUS,
                    nb_cpu=self.nb_cpu
                )
                job = year_study.run_antares_simulation(parameters=params)
                self.jobs[year] = {"job": job, "study": year_study}
                print(f"Job '{job.job_id}' submitted for year {year}.")

            except Exception as e:
                print(f"!! FATAL ERROR processing year {year}: {e}")
                self.jobs[year] = {"job": None, "error": str(e)}

        return definitive_study_ids

    def wait_for_completion(self, timeout_sec: int = 3600):
        """
        Waits for all submitted jobs to complete by systematically polling and
        updating their status, following a clear batch-update logic.
        """
        print("\n--- Waiting for All Jobs to Complete ---")
        start_time = time.time()

        years_with_jobs = list(self.jobs.keys())
        if not years_with_jobs:
            print("No jobs to wait for.")
            return

        while True:
            # 1. Check for timeout
            if time.time() - start_time > timeout_sec:
                print("\n!! ERROR: Timeout reached while waiting for jobs to complete.")
                # Mark any remaining running/pending jobs as failed
                for year in years_with_jobs:
                    job = self.jobs[year].get("job")
                    if job and job.status in [ac.model.simulation.JobStatus.RUNNING, ac.model.simulation.JobStatus.PENDING]:
                        self.jobs[year]["job"].status.value = ac.model.simulation.JobStatus.FAILED
                        self.jobs[year]["error"] = "Timeout"
                break

            print(f"\n{datetime.datetime.now().strftime('%H:%M:%S')}: Refreshing job statuses...")

            # 2. Systematically poll and update every job object
            for year in years_with_jobs:
                job_info = self.jobs[year]
                job = job_info.get("job")
                study = job_info.get("study")

                if not job or not study:
                    continue  # Skip years where job submission failed

                # Update only if not already in a terminal state
                if job.status not in [ac.model.simulation.JobStatus.SUCCESS, ac.model.simulation.JobStatus.FAILED]:
                    try:
                        updated_job = study._run_service._get_job_from_id(job.job_id, job.parameters)
                        self.jobs[year]["job"] = updated_job
                    except Exception as e:
                        print(f"!! Could not get status for job '{job.job_id}'. Marking as FAILED. Error: {e}")
                        self.jobs[year]["job"].status = ac.model.simulation.JobStatus.FAILED
                        self.jobs[year]["error"] = str(e)

            # 3. Check for completion by counting finished jobs
            successful_years = [y for y, info in self.jobs.items() if
                                info.get("job") and info["job"].status == ac.model.simulation.JobStatus.SUCCESS]
            failed_years = [y for y, info in self.jobs.items() if
                            info.get("job") and info["job"].status == ac.model.simulation.JobStatus.FAILED]

            print(
                f"  - Status: {len(successful_years)} Succeeded, {len(failed_years)} Failed, {len(years_with_jobs) - len(successful_years) - len(failed_years)} Running/Pending")

            if len(successful_years) + len(failed_years) == len(years_with_jobs):
                print("\nAll jobs have completed.")
                break

            # Wait before the next polling cycle
            time.sleep(10)

        # --- Final Summary ---
        successful_years = [y for y, info in self.jobs.items() if
                            info.get("job") and info["job"].status == ac.model.simulation.JobStatus.SUCCESS]
        failed_years = [y for y, info in self.jobs.items() if
                        not info.get("job") or info["job"].status != ac.model.simulation.JobStatus.SUCCESS]

        print("-" * 40)
        if not failed_years:
            print("\033[1mAll jobs succeeded!\033[0m")
        else:
            print("\033[1mBatch finished with some failures.\033[0m")
        print(f"  - Successful years: {successful_years}")
        print(f"  - Failed years: {failed_years}")
        print("-" * 40)

    def get_all_results(self) -> pd.DataFrame:
        """Collects adequacy metrics from all successful jobs in the batch."""
        print("\n--- Collecting Results from All Successful Jobs ---")
        all_results = []

        for year, job_info in self.jobs.items():
            job = job_info.get("job")
            study = job_info.get("study")  # <-- THE FIX: Access the stored study object

            if not job or not study or job.status != ac.model.simulation.JobStatus.SUCCESS or not job.output_id:
                if job:
                    print(f"Skipping results for year {year} due to simulation status: {job.status.value}")
                continue

            try:
                # Use the stored study object to get results
                reader = AntaresResultsReader(study, job.output_id)
                df = reader.get_adequacy_metrics()
                df['year'] = year
                all_results.append(df)
            except Exception as e:
                print(f"Could not retrieve results for successful job of study '{study.name}'. Error: {e}")

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results).reset_index().rename(columns={"index": "area"})