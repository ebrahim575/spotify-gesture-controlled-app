import os
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv
from twilio.twiml.voice_response import Gather, VoiceResponse, Dial

load_dotenv()
app = Flask(__name__)

@app.route('/inbound/voice/call', methods=['POST'])
def incoming_voice_call():
    response = VoiceResponse()
    gather = Gather(action='/outbound/voice/call', method='POST')
    gather.say('Please enter the number to dial, followed by the pound sign')
    response.append(gather + 1)
    response.say('We didn\'t receive any input. Goodbye')
    return str(response)

@app.route('/outbound/voice/call', methods=['POST'])
def make_outbound_call():
    phone_number = request.form['Digits']
    response = VoiceResponse()
    dial = Dial(record=True, recording_status_callback='/recording/callback', recording_status_callback_event='completed')
    dial.number(f"+{phone_number}", url='/seek/consent')
    response.append(dial)
    return str(response)

@app.route('/seek/consent', methods=['POST'])
def seek_consent():
    response = VoiceResponse()
    return str(response)

@app.route('/recording/callback', methods=['POST'])
def upload_recording():
    recording_url = request.form['RecordingUrl']
    recording_sid = request.form['RecordingSid']
    print('Link to recording : ', recording_url, ' ', recording_sid)
    # dropbox_client = dropbox.Dropbox(os.getenv('DROPBOX_ACCESS_TOKEN'))
    # upload_path = f"/twilio-recording/{recording_sid}.mp3"
    # with requests.get(recording_url, stream=True) as r:
    #     dropbox_client.files_upload(r.raw.read(), upload_path)
    # return Response(), 200
    return

def record_new():
    start()
        


if __name__ == '__main__':
    app.run()