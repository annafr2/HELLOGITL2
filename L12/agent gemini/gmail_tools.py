import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailTools:
    def __init__(self):
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'json/credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
    
    def list_labels(self):
        """Get list of all Gmail labels"""
        labels = self.service.users().labels().list(userId='me').execute()
        return [label['name'] for label in labels.get('labels', [])]
    
    def search_emails_by_label(self, label_name):
        """Search emails by label"""
        labels = self.service.users().labels().list(userId='me').execute()
        label_id = None
        
        for label in labels.get('labels', []):
            if label['name'].upper() == label_name.upper():
                label_id = label['id']
                break
        
        if not label_id:
            return {"error": f"Label '{label_name}' not found"}
        
        results = self.service.users().messages().list(
            userId='me',
            labelIds=[label_id],
            maxResults=100
        ).execute()
        
        messages = results.get('messages', [])
        
        emails_data = []
        for message in messages:
            email_details = self.get_email_details(message['id'])
            if email_details:
                emails_data.append(email_details)
        
        return emails_data
    
    def get_email_details(self, message_id):
        """Get email details"""
        message = self.service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()
        
        headers = message['payload']['headers']
        
        email_data = {
            'id': message_id,
            'date': '',
            'from': '',
            'subject': '',
            'snippet': message.get('snippet', '')
        }
        
        for header in headers:
            name = header['name'].lower()
            if name == 'date':
                email_data['date'] = header['value']
            elif name == 'from':
                email_data['from'] = header['value']
            elif name == 'subject':
                email_data['subject'] = header['value']
        
        return email_data
    
    def save_to_excel(self, emails_data, label_name):
        """Save emails to Excel file"""
        if not emails_data:
            return {"error": "No emails to save"}
        
        df = pd.DataFrame(emails_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{label_name.lower()}_emails_{timestamp}.xlsx'
        df.to_excel(filename, index=False, engine='openpyxl')
        
        return {
            "success": True,
            "filename": filename,
            "count": len(emails_data)
        }


# Create global instance
gmail = GmailTools()


def get_labels():
    """Returns list of labels"""
    return {"labels": gmail.list_labels()}


def search_and_save_emails(label_name):
    """Search and save emails by label"""
    emails = gmail.search_emails_by_label(label_name)
    
    if isinstance(emails, dict) and "error" in emails:
        return emails
    
    result = gmail.save_to_excel(emails, label_name)
    return result