'''
'''
import os
import sys


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from automated_training.auxiliary_functions import setup_gpus, train_save_evaluate_model, apply_exclusions
from automated_training.exit_codes import SWITCH_TRAFFIC_EXIT_CODE, DO_NOT_SWITCH_TRAFFIC_EXIT_CODE

from ficc.utils.auxiliary_functions import function_timer


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    return train_save_evaluate_model('yield_spread_with_similar_trades', apply_exclusions, current_date_passed_in, True)


if __name__ == '__main__':
    setup_gpus()
    switch_traffic = main()
    if switch_traffic:
        print(f'Switching traffic so returning an exit code of {SWITCH_TRAFFIC_EXIT_CODE} to be handled in the bash script')
        sys.exit(SWITCH_TRAFFIC_EXIT_CODE)
    else:
        print(f'NOT Switching traffic so returning an exit code of {DO_NOT_SWITCH_TRAFFIC_EXIT_CODE} to be handled in the bash script')
        sys.exit(DO_NOT_SWITCH_TRAFFIC_EXIT_CODE)
