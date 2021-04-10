import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from jammy.logging import get_logger
from .env import jam_getenv
from .meta import run_once

logger = get_logger()

__all__ = ["jam_notifier", "get_notifier"]


class _Notifier:
    def __init__(self):
        self.enabled = True
        keys = ["sender", "receiver", "smtp_url", "smtp_port", "smtp_key"]
        for cur_key in keys:
            term = jam_getenv(f"ntf_{cur_key}", default=None, prefix="_")
            if term is None:
                self.enabled = False
                logger.debug(f"miss {cur_key} in environment")
                break
            setattr(self, cur_key, term)
        try:
            user = os.getlogin()
        except:
            user = "qzhang419"
        self.address = f"{user}@{os.uname().nodename}"
        self.user = user

        self.server = smtplib.SMTP(self.smtp_url, self.smtp_port)
        self.server.login(self.sender, self.smtp_key)

    @run_once
    def warn(self):
        logger.critical("Notifier not setup")

    def notify(self, msg, subject=None):
        if not self.enabled:
            self.warn()
            return False
        logger.debug(f"send {msg}")
        try:
            msg = MIMEText(msg, "plain", "utf-8")
            msg["From"] = formataddr([f"NotifierEXP {self.address}", self.sender])
            msg["To"] = formataddr([f"{self.user}", self.receiver])
            if subject == None:
                subject = "Update on EXP"
            msg["Subject"] = subject

            self.server.sendmail(
                self.sender,
                [
                    self.receiver,
                ],
                msg.as_string(),
            )
            return True
        except Exception as e:
            logger.warning(str(e))
            return False

        def __exit__(self):
            self.server.quit()


jam_notifier = _Notifier()


def get_notifier():
    return jam_notifier
