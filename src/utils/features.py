import pandas as pd

import pandas as pd


class TimeFeatures:
    @staticmethod
    def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'date' column.

        Returns:
            pd.DataFrame: The DataFrame with added time-based features.

        Note:
        - The 'date' column in the DataFrame should be of type pandas datetime.
        - The 'count' column is assumed to be present in the DataFrame and will be included in the final DataFrame.

        Example Usage:
        >>> df = pd.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03'], 'count': [10, 20, 30]})
        >>> TimeFeatures.make_time_features(df)
        """
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month_of_year'] = df['date'].dt.month
        # Compute quarter of the year
        df['quarter'] = df['date'].dt.quarter
        # Compute the number of days in the month
        df['days_in_month'] = df['date'].dt.daysinmonth
        # Distance from 2021-01-01
        df['days_from_start'] = (
            df['date'] - pd.to_datetime('2021-01-01')).dt.days
        # Reorder the columns
        df = df[[
            'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month_of_year',
            'quarter', 'days_in_month',
            'days_from_start', 'count']]

        return df
