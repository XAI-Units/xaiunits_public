import pickle
import pprint
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Any, Callable, Dict, Union


@dataclass(order=True)
class Example:
    """
    Stores a datapoint with its attributions and score, for ranking top n examples.
    """

    score_for_ranking: float = field(compare=True, init=False, repr=False)
    score: float = field(compare=False)
    attribute: Tensor = field(compare=False)
    feature_inputs: Tensor = field(compare=False)
    y_labels: Tensor = field(compare=False)
    target: int | None = field(compare=False)
    context: dict | None = field(compare=False)
    example_type: str = field(compare=False)

    def __post_init__(self) -> None:
        """
        Adjusts the `score_for_ranking` to be negative for "min" examples.
        This is so that we can use heapq, a min heap, as a max heap.

        Raises:
            ValueError: If the value of example_type is not accepted.
        """
        if self.example_type == "max":
            self.score_for_ranking = self.score
        elif self.example_type == "min":
            self.score_for_ranking = -self.score
        else:
            raise ValueError("example_type must be 'max' or 'min'")


class Results:
    """
    Object that records and processes the results of experiments from a subclass of BasePipeline.

    Attributes:
        data (list): List of all stored experiment results from a pipeline object.
        metrics (list): List of the names of all evaluation metrics recorded in data.
        examples (dict): Dictionary containing the data samples that output the maximum or minimum
            scores with respect to the evaluation metrics of interest.
    """

    def __init__(self) -> None:
        """Initializes a Results object."""
        self.raw_data = []
        self.examples = {"max": defaultdict(list), "min": defaultdict(list)}

    def append(self, incoming):
        """
        Appends the new result to the collection of results stored.

        Args:
            incoming (dict): New result to be appended to the existing results.
        """
        self.raw_data.append(incoming)

    @property
    def data(self) -> pd.DataFrame:
        """
        Processes the raw_data list into a DataFrame, flattening over the batch dimension.

        For each data instance, the value under the 'attr_time' column will be the time it
        took for the batch, it belongs to, to compute its respective attribution scores,
        divided by the size of the batch.

        Returns:
            pandas.DataFrame: The processed results, one row per datapoint.
        """
        batch_columns = ["batch_row_id", "value"]
        df_1row_per_batch = pd.DataFrame(self.raw_data)

        # divide the attribution time by the batch size
        divide_by_batch_size = lambda row: row["attr_time"] / len(row["batch_row_id"])
        df_1row_per_batch["attr_time"] = df_1row_per_batch.apply(
            divide_by_batch_size, axis=1
        )

        # explode flattens out the dataframe
        # so instead of one row per batch it becomes one row per datapoint
        data = df_1row_per_batch.explode(batch_columns, ignore_index=True)

        # remove the tensor wrapping of each cell
        extract_tensor_value = lambda t: t.item() if isinstance(t, torch.Tensor) else t
        data = data.map(extract_tensor_value)

        return data

    def process_data(self) -> None:
        """
        Convenience method for accessing the self.data property.

        Returns:
            pandas.DataFrame: The processed results, one row per datapoint.
        """
        return self.data

    def print_stats(
        self,
        metrics: Optional[List[Any]] = None,
        stat_funcs: List[str] = ["mean", "std"],
        index: List[str] = ["data", "model", "method"],
        initial_mean_agg: List[str] = ["batch_id", "batch_row_id"],
        time_unit_measured: str = "dataset",
        decimal_places: int = 3,
        column_index: List = [],
    ) -> pd.DataFrame:
        """
        Prints the results in the form of a pivot table for each of the statistic required.

        The indices of the printed table correspond to the model and explanation method,
        and the columns of the printed table correpond to the evaluation metrics. The values
        of the printed table are the statistics calculated across the number of experiment trials.
        When mean and standard deviation are needed, a single pivot table will be printed that
        records both of them.

        Args:
            metrics (list, optional): A list of the names of all metrics to be printed. Defaults to None.
            stat_funcs (list | str): A list of aggregation functions that are required to be printed.
                It can be a str if only a single type of aggregation is required. Supports the same
                preset and custom aggregation functions as supported by pandas.
                Defaults to ['mean', 'std'].
            index (list): A list of the names of the columns to be used as indices in the pivot table.
                Defaults to ['data', 'model', 'method']
            initial_mean_agg (list): A list of the columns to be used for the initial mean.
                For example, if we want to calculate the standard deviation between experiments,
                we take the mean over batch_id and batch_row_id first, then calculate the
                standard deviation. Defaults to ['batch_id', 'batch_row_id'].
            time_unit_measured (str): The unit for which the time to perform the attribution method is
                calculated and aggregated. Defaults to 'dataset', and only 3 values are supported inputs:
                - 'dataset': time needed to apply explanation method on each dataset
                - 'batch': time needed to apply explanation method on each batch
                - 'instance': time needed to apply explanation method on each data instance. It is
                estimate derived from the batch time.
            decimal_places (int): The decimal places of the values displayed. Defaults to 3
                decimal places.
            column_index (list): A list of column names to unpivot, in addition to Stats and Metrics.
                Defaults to [], so just Stats and Metrics are used as columns.
        """
        # modify the dataset such that the attribution time is expressed in the correct unit

        attr_time_aggregated_df = self._attr_time_summing(time_unit_measured)

        # partition the dataset by metrics wanted
        if metrics is not None:
            partitioned_data = attr_time_aggregated_df[
                attr_time_aggregated_df["metric"].isin(metrics)
            ]
        else:
            partitioned_data = attr_time_aggregated_df

        # determine the columns to preserve over the initial mean aggregation
        if initial_mean_agg:
            cols_to_preserve = list(
                set(partitioned_data.columns) - set(initial_mean_agg) - {"value"}
            )

            # perform mean aggregate over experiments / batches
            results_by_experiment = pd.pivot_table(
                partitioned_data,
                values="value",
                index=cols_to_preserve,
                dropna=False,
            )
        else:
            results_by_experiment = partitioned_data

        # perform aggregation of statistic of interest over the measured metrics
        if "metric" not in column_index:
            column_index = column_index + ["metric"]
        stat_table = pd.pivot_table(
            results_by_experiment,
            values="value",
            index=index,
            aggfunc=stat_funcs,
            dropna=False,
            columns=column_index,
        )

        # drop rows that contains all NaNs
        stat_table.dropna(how="all", inplace=True)

        n_cols = len(stat_table.columns)
        with pd.option_context(
            "display.precision", decimal_places, "display.max_columns", n_cols
        ):
            print(stat_table)

        return stat_table

    def print_all_results(self) -> None:
        """Prints all the data as a wide table."""
        pprint.pprint(self.data)

    def save(self, filepath):
        """
        Saves the data stored as a .pkl file.

        Args:
            filepath (str): Path for the .pkl file to be saved in.
        """
        with open(filepath, "wb") as file:
            pickle.dump(self.raw_data, file)

    def load(self, filepath: str, overwrite: bool = False) -> None:
        """
        Loads the data from a .pkl file.

        The data loaded will be concatenated with the existing data store in the object
        if and only if overwrite is False.

        Args:
            filepath (str): Path of the .pkl file to load the data from.
            overwrite (bool): True if and only if the data loaded overwrites any
                existing data stored. Defaults to False.
        """
        with open(filepath, "rb") as file:
            data_loaded = pickle.load(file)

        if overwrite:
            self.raw_data = data_loaded
        else:
            self.raw_data += data_loaded

    def print_max_examples(self) -> None:
        """
        Prints in descending order the collection of examples that give the maximum
        evaluation metric scores.
        """
        print("Max Scoring Examples")
        max_examples = self.examples["max"]
        max_examples_descending = {k: v[::-1] for k, v in max_examples.items()}
        pprint.pprint(max_examples_descending)

    def print_min_examples(self) -> None:
        """
        Prints in ascending order the collection of examples that give the minimum
        evaluation metric scores.
        """
        print("Min Scoring Examples")
        min_examples = self.examples["min"]
        min_examples_ascending = {k: v[::-1] for k, v in min_examples.items()}
        pprint.pprint(min_examples_ascending)

    def _attr_time_summing(self, time_unit_measured: str) -> pd.DataFrame:
        """
        Aggregates attribution time based on the specified time unit and formats the data
        in a correct format with the aggregated attribution times as part of the metrics.

        Args:
            time_unit_measured (str): The unit for which the time to perform the
                attribution method is calculated and aggregated. Only allows
                inputs "dataset", "batch", "instance".

        Returns:
            pandas.DataFrame: pandas.DataFrame object with aggregated attribution time results
                formatted correctly.

        Raises:
            Exception: If an invalid time unit is provided.
        """
        if time_unit_measured == "dataset":
            # aggregates attritbution time over each dataset
            time_df = self.data.drop(columns=["value", "metric"])
            time_agg_cols = ["batch_id", "batch_row_id", "attr_time"]
            time_cols = [col for col in time_df.columns if col not in time_agg_cols]

            attr_time_summed = pd.pivot_table(
                time_df,
                values="attr_time",
                index=time_cols,
                aggfunc="sum",
                dropna=False,
            )

            attr_time_summed = pd.DataFrame(attr_time_summed.to_records())
            attr_time_summed["batch_row_id"] = 0
            attr_time_summed["batch_id"] = 0

        elif time_unit_measured == "batch":
            # aggregates attritbution time over each batch
            time_df = self.data.drop(columns=["value", "metric"])
            time_agg_cols = ["batch_row_id", "attr_time"]
            time_cols = [col for col in time_df.columns if col not in time_agg_cols]

            attr_time_summed = pd.pivot_table(
                time_df,
                values="attr_time",
                index=time_cols,
                aggfunc="sum",
                dropna=False,
            )

            attr_time_summed = pd.DataFrame(attr_time_summed.to_records())
            attr_time_summed["batch_row_id"] = 0

        elif time_unit_measured == "instance":
            # aggregates attritbution time over each data instance
            attr_time_summed = self.data.drop(columns=["metric", "value"])
        else:
            raise Exception(
                "Invalid unit of attribution time inputed. Only 'dataset', 'batch', 'instance' are accepted."
            )

        # rename columns and merge with original dataset
        attr_time_summed["metric"] = "attr_time"
        attr_time_summed = attr_time_summed.rename(columns={"attr_time": "value"})
        no_time_df = self.data.drop(columns=["attr_time"])
        return pd.concat([no_time_df, attr_time_summed], ignore_index=True)
