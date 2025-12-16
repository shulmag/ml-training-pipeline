'''
 # creatd in order to separate the complex logic of `trade_dict_to_list.py` from these auxiliary 
 # variables.
 '''
def list_to_index_dict(lst):
    '''Returns a dictionary from a list, where the key is the index and the value is 
    the corresponding item in the list. Works when `lst` is a numpy array as well.'''
    return {idx: item for idx, item in enumerate(lst)}


TRADE_TYPE_MAPPING = {'D': [1, 1],
                      'S': [0, 1], 
                      'P': [1, 0], 
                      'E': [0, 0]}    # E represents empty, for when we have empty (i.e., padded) trades in the trade history


TRADE_TYPE_CROSS_PRODUCT_MAPPING = {'Empty': 0,    # 'Empty' represents empty, for when we have empty (i.e., padded) trades in the trade history
                                    'DD': 1, 
                                    'DP': 2, 
                                    'DS': 3, 
                                    'PD': 4, 
                                    'PP': 5, 
                                    'PS': 6, 
                                    'SD': 7, 
                                    'SP': 8, 
                                    'SS': 9}


RATINGS = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-']    # hard coded list allows for ordering based on meaning of the rating
RATING_TO_INT_MAPPING = list_to_index_dict(RATINGS)    # | {'MR': -1, 'NR': -1}     # combine two dictionaries together for Python v3.9+: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression
RATING_TO_INT_MAPPING['MR'] = -1
RATING_TO_INT_MAPPING['NR'] = -1
