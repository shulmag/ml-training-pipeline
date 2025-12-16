'''
'''
import numpy as np

from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.nelson_siegel_model import get_duration


def get_treasury_rate_dict(client) -> dict:
    '''Get the treasury rate data from BigQuery and return it as a dictionary. The output dictionary is structured as:
    {
        datetime.date(2025, 5, 8): {'year_1': 4.05, 'year_2': 3.9, 'year_3': 3.85, 'year_5': 4.0, 'year_7': 4.18, 'year_10': 4.37, 'year_20': 4.86, 'year_30': 4.83}, 
        datetime.date(2025, 5, 7): {'year_1': 4.0, 'year_2': 3.78, 'year_3': 3.72, 'year_5': 3.87, 'year_7': 4.06, 'year_10': 4.26, 'year_20': 4.78, 'year_30': 4.77},
        ...
    }'''
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc'''
    treasury_rate_df = sqltodf(query, client)
    treasury_rate_df = treasury_rate_df.drop_duplicates(keep='first')    # from testing or manual corrections, sometimes there are duplicate entries in the table
    treasury_rate_df.set_index('Date', drop=True, inplace=True)
    return treasury_rate_df.transpose().to_dict()


def current_treasury_rate(treasury_rate_dict: dict, trade):
    '''If the trade date corresponding to `trade` is not found in `treasury_rate_dict`, 
    then return np.nan. This is later filtered out in `process_data(...)`.'''
    trade_date = trade['trade_date'].date()    # converts pd.Timestamp (original version of `trade['trade_date']`) to datetime.date since this is the type of the keys in `treasury_rate_dict`
    if trade_date not in treasury_rate_dict: return np.nan
    treasury_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
    time_to_maturity = get_duration(trade, use_last_calc_day_cat=False)
    maturity = treasury_maturities[np.argmin(np.abs(treasury_maturities - time_to_maturity))]    # faster than using `min(...)` (e.g., `min(treasury_maturities, key=lambda treasury_maturity: abs(treasury_maturity - time_to_maturity))`) because numpy vectorizes the operation
    return treasury_rate_dict[trade_date][f'year_{maturity}']
