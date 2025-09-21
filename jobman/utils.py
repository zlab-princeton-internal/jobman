import os
import sys
import logging
from pathlib import Path

def setup_logger(name: str = 'job', log_file: Path = None, level=logging.DEBUG, stdout=False):
    """Configure global logging to stdout and/or a log file."""
    # Get root logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if setup_logger is called more than once
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="(%(asctime)s) [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Optional terminal output
    if stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Optional file output
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from sib_api_v3_sdk import  ApiClient, Configuration, TransactionalEmailsApi, SendSmtpEmail
from pprint import pprint

def send_notification(msg, config, title="Jobman Notification"):
    if os.environ.get("DISABLE_EMAIL", "").lower() in ("1", "true", "yes", "on"):
        return
    
    configuration = sib_api_v3_sdk.Configuration()
    sender = config.get('brevo_email', {}).get('sender', None)
    receiver = config.get('brevo_email', {}).get('receiver', None)
    configuration.api_key['api-key'] = config.get('brevo_email', {}).get('api_key', None)
    
    if not sender or not receiver or not configuration.api_key['api-key']:
        return

    api_client = sib_api_v3_sdk.ApiClient(configuration)

    api_instance = TransactionalEmailsApi(ApiClient(configuration))
    send_email = SendSmtpEmail(
        to=[{"email": receiver, "name": "Jobman User"}],
        sender={"email": sender, "name": "Jobman Bot"},
        subject=title,
        text_content=msg,
    )

    try:
        api_response = api_instance.send_transac_email(send_email)
        print("✅ Campaign created successfully:")
        pprint(api_response)
    except ApiException as e:
        print("❌ Exception when calling EmailCampaignsApi->create_email_campaign:\n%s" % e)