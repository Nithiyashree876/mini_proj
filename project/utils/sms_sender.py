"""
SMS Notification Module.
Simulates sending a text message using a mock setup, 
with built-in stubs ready for Twilio integration.
"""

class SMSSender:
    def __init__(self):
        # To use real SMS, fill these in with Twilio credentials:
        self.account_sid = "TWILIO_ACCOUNT_SID_HERE"
        self.auth_token = "TWILIO_AUTH_TOKEN_HERE"
        self.from_phone = "+1234567890"

    def send_sms(self, to_phone, message):
        """Mock SMS sending function."""
        if not to_phone:
            return False
            
        print("\n" + "="*50)
        print("📱 [SIMULATED SMS SENT]")
        print(f"To: {to_phone}")
        print(f"Message: {message}")
        print("="*50 + "\n")
        
        # NOTE: To use real Twilio, uncomment and pip install twilio:
        # from twilio.rest import Client
        # try:
        #     client = Client(self.account_sid, self.auth_token)
        #     client.messages.create(
        #         body=message,
        #         from_=self.from_phone,
        #         to=to_phone
        #     )
        #     return True
        # except Exception as e:
        #     print(f"SMS Failed: {e}")
        #     return False
            
        return True
