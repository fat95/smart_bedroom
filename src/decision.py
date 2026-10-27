# src/decision.py
from config import DEVICES, ENROLLED_SPEAKERS, SPEAKER_VERIFICATION_THRESHOLD

def decide(command_data, speaker_data):
    if not command_data or not command_data.get("device"):
        return {"status": "rejected", "reason": "no_speech"}
        
    device = command_data.get("device")
    action = command_data.get("action")
    value = command_data.get("value")
    
    if device not in DEVICES:
        return {"status": "rejected", "reason": "unknown_command"}
        
    if value is not None:
        try:
            val = float(value)
            if val < DEVICES[device]["min"] or val > DEVICES[device]["max"]:
                return {"status": "rejected", "reason": "invalid_value"}
        except ValueError:
            return {"status": "rejected", "reason": "invalid_value"}

    if not speaker_data:
        return {"status": "rejected", "reason": "unauthorized_speaker"}
        
    score = speaker_data.get("score", 0)
    speaker_id = speaker_data.get("speaker_id")
    
    if score < SPEAKER_VERIFICATION_THRESHOLD or speaker_id not in ENROLLED_SPEAKERS:
        return {"status": "rejected", "reason": "unauthorized_speaker"}
        
    return {
        "status": "ok", 
        "device": device, 
        "action": action, 
        "value": value,
        "speaker": speaker_id
    }