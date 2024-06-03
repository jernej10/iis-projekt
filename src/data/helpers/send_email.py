import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

load_dotenv()

def send_email(text: str):

    sender_email = os.getenv("GMAIL")
    password = os.getenv("GMAIL_PASSWORD")
    receiver_email = os.getenv("GMAIL_PASSWORD")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "ALERT - production model evaluation"

    body = text
    message.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

    print("Email sent successfully!")

if __name__ == "__main__":
    send_email()
