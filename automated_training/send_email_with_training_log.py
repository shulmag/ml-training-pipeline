'''
Description: Used to send an email to a list of recipients with the training log attached. This allows the the recipients to view the training log without having to ssh into the VM where the training occurs.
'''
import os
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from automated_training.auxiliary_variables import EMAIL_RECIPIENTS_FOR_LOGS, BUCKET_NAME, TRAINING_LOGS_DIRECTORY
from automated_training.auxiliary_functions import send_email, check_that_model_is_supported, STORAGE_CLIENT


from ficc.utils.gcp_storage_functions import upload_data


SUCCESS_MESSAGE = 'No detected errors. Logs attached for reference.'    # must match the message at the bottom of bash script: `model_deployment.sh`


def get_filename_from_path(path: str) -> str:
    '''Find the last occurrence of a slash and then keep the rest of the string.'''
    last_slash_index = path.rfind('/')    # rfind starts searching from the right side (hence the 'r' in `rfind`) and returns the index of the last occurrence of the specified substring
    assert last_slash_index != -1, f'No "/" found in {path}'    # if '/' is not found, then `rfind(...)` returns -1
    return path[last_slash_index + 1:]


def store_logs_in_google_cloud_storage(file_name: str, file_path: str, model: str):
    '''Store the logs in Google Cloud Storage in bucket `BUCKET_NAME` and directory `TRAINING_LOGS_DIRECTORY`.'''
    print(f'Storing logs in Google Cloud Storage for {model} model')
    upload_data(STORAGE_CLIENT, BUCKET_NAME, f'{TRAINING_LOGS_DIRECTORY}/{model}/{file_name}', file_path)


def send_training_log(attachment_path, recipients: list, model: str, message: str):
    check_that_model_is_supported(model)

    attachment_filename = get_filename_from_path(attachment_path)
    if message != SUCCESS_MESSAGE:    # only send an email with logs if there is an error
        print(f'Error detected in training. Sending email to {recipients}')
        sender_email = 'notifications@ficc.ai'
        
        msg = MIMEMultipart()
        msg['Subject'] = f'(v2) Training log for {model} model trained today'
        msg['From'] = sender_email

        body = MIMEText(message, 'plain')
        msg.attach(body)

        with open(attachment_path, 'rb') as attachment:    # attach the file
            part = MIMEApplication(attachment.read())
            part.add_header('Content-Disposition', 'attachment', filename=attachment_filename)
            msg.attach(part)
        send_email(sender_email, msg, recipients)

    store_logs_in_google_cloud_storage(attachment_filename, attachment_path, model)    # run this regardless of whether there is an error or not


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: $ python send_email_with_training_log.py <filepath> <model> <message>')
    else:
        filepath = sys.argv[1]
        model = sys.argv[2]
        message = sys.argv[3]
        print(f'Sending email with {filepath}')
        send_training_log(filepath, EMAIL_RECIPIENTS_FOR_LOGS, model, message)
