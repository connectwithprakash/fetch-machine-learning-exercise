import pandas as pd

class TimeFeatures:
    @staticmethod
    def make_time_features(df):
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
        df['days_from_start'] = (df['date'] - pd.to_datetime('2021-01-01')).dt.days
        # Reorder the columns
        df = df[[
            'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month_of_year', 
            'quarter', 'days_in_month',
            'days_from_start', 'count']]

        return df
