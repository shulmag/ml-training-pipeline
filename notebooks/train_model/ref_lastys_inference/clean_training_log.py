'''
'''
import sys


def remove_lines_with_character(character_to_remove, file_path: str, new_file_path: str = None):
    with open(file_path, 'r') as file:    # read the file
        lines = file.readlines()
    filtered_lines = [line for line in lines if character_to_remove not in line]    # filter out lines containing the specified character
    if new_file_path is None: new_file_path = file_path
    with open(new_file_path, 'w') as file:    # write the filtered lines back to the file
        file.writelines(filtered_lines)


def remove_lines_with_tensorflow_progress_bar(file_path: str):
    remove_lines_with_character('', file_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: $ python clean_training_log.py <file_path>')
    else:
        file_path = sys.argv[1]
        print(f'Cleaning up: {file_path}')
        remove_lines_with_tensorflow_progress_bar(file_path)