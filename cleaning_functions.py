import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Helpful functions for data cleaning


def combine_csv(csv_file_path: list):
    """
    This function imports csv files and
    combines them into a single DataFrame
    :param: List of file paths to csvs
    :return: Concatenated DataFrame
    """

    csv_files = [pd.read_csv(f, index_col=0, usecols=range(1, 20)) for f in csv_file_path]

    return pd.concat(csv_files)


class CleanSkiData:
    """
    Contains a number of helpful functions to clean the data
    extracted from skiresort.info using the main.py script
    """
    def __init__(self, dataframe):
        self.data = dataframe

    def check_null_values(self):
        # returns a breakdown of null values in each column
        return self.data.isnull().mean() * 100

    def drop_empty_cost_rows(self, columns="Ski Pass Cost"):
        # For a given column, drops any rows with null values
        # Default is Ski Pass Cost, because it is a key value for the project
        # inplace is set to False by default so the dataframe is returned
        if type(columns) is not list:
            columns = [columns]

        self.data.dropna(subset=columns, inplace=True)

    def split_cost_columns(self):
        # Split out the cost into 3 columns
        # "Currency"
        # "Ski Pass Cost"
        # "Cost in Euros"

        # Step 1 - Replace ",-" and " / approx."
        self.data["Ski Pass Cost"] = self.data["Ski Pass Cost"].str.replace(",-", "", regex=False)
        self.data["Ski Pass Cost"] = self.data["Ski Pass Cost"].str.replace(" / approx.", "", regex=False)

        # Split the strings
        self.data["Ski Pass Cost"] = self.data["Ski Pass Cost"].str.split()

        # Now create the new columns
        # The first value in the list is the currency
        # The second value the original cost
        # The -1 value shall be the cost in euros
        self.data["Currency"] = self.data["Ski Pass Cost"].str[0]
        self.data["Cost in Euros"] = self.data["Ski Pass Cost"].str[-1]
        self.data["Ski Pass Cost"] = self.data["Ski Pass Cost"].str[1]

    def make_values_numerical(self, columns):
        # Where data in a column includes a unit this is removed
        # Leaves only the numerical values
        if type(columns) is not list:
            columns = [columns]

        for column in columns:
            self.data[column] = self.data[column].str.split().str[0]

    def clean_resort_names(self):
        # Removes the phrase "temporarily closed" from ski resort names
        # This information is unnecessary
        self.data["Name"] = self.data["Name"].str.replace(" (temporarily closed)", "", regex=False)

    def check_unique(self):
        # Confirms there are no duplicate resorts
        if len(self.data["Name"].unique()) == len(self.data["Name"]):
            return "No duplicate resorts!"
        else:
            return "There are duplicate resorts!"
