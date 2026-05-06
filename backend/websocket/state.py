class SessionState:
    def __init__(self):
        self.client_sample_rate = 48000  # default fallback
        self.reset()
    
    def reset(self):
        self.buf = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.active = False
        self.last_speech_time = None
        self.processing = False
        self.cancel_current = False
