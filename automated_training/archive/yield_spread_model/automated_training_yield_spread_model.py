'''
'''
import sys
from ficc.utils.auxiliary_functions import function_timer

from auxiliary_functions import setup_gpus, train_save_evaluate_model, apply_exclusions


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('yield_spread', apply_exclusions, current_date_passed_in, True)


if __name__ == '__main__':
    setup_gpus()
    main()
