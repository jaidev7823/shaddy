class SessionState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.buf = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.active = False
        self.last_speech_time = None
