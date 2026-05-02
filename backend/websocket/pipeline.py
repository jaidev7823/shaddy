import time
from typing import Optional, Dict, Any
from backend.websocket.cooldown import cooldown_manager

class Pipeline:
    def __init__(self, speaker_service, transcription_service, llm_service, tts_service):
        self.speaker_service = speaker_service
        self.transcription_service = transcription_service
        self.llm_service = llm_service
        self.tts_service = tts_service
    
    def process_utterance(self, full_audio: bytes) -> Dict[str, Any]:
        times = {}
        t_total = time.perf_counter()

        result = {
            "is_student": False,
            "transcript": None,
            "similarity": None,
            "llm_result": None,
            "response_data": {},
            "should_nudge": False,
            "nudge_text": None,
            "audio_generated": False,
            "cooldown_active": False,
        }
        
        t = time.perf_counter()
        result["is_student"] = self.speaker_service.is_student_voice(full_audio)
        times["speechbrain"] = round(time.perf_counter() - t, 3)

        if result["is_student"]:
            print(f"  ⏱ speechbrain={times['speechbrain']}s — student voice, skipped")
            return result
        
        t = time.perf_counter()
        result["transcript"] = self.transcription_service.transcribe_from_bytes(full_audio)
        times["whisper"] = round(time.perf_counter() - t, 3)

        if not result["transcript"]:
            print(f"  ⏱ speechbrain={times['speechbrain']}s | whisper={times['whisper']}s — no transcript")
            return result
        
        print(f'  heard: "{result["transcript"]}"')

        t = time.perf_counter()
        result["similarity"] = self.speaker_service.get_speaker_similarity(full_audio)
        times["similarity"] = round(time.perf_counter() - t, 3)

        result["llm_result"] = None
        t = time.perf_counter()
        result["llm_result"] = self.llm_service.process_transcript(result["transcript"])
        times["llm"] = round(time.perf_counter() - t, 3)
        
        response_data = {
            "transcript": result["transcript"],
            "speaker_similarity": result["similarity"],
            "llm_response": result["llm_result"],
        }
        
        lesson_id = result["llm_result"].get("lesson_id")
        if lesson_id:
            if cooldown_manager.is_active(lesson_id):
                response_data["cooldown_active"] = True
                response_data["should_nudge"] = False
                result["cooldown_active"] = True
                result["response_data"] = response_data
                print(f"  ⏱ speechbrain={times['speechbrain']}s | whisper={times['whisper']}s | llm={times['llm']}s — cooldown active")
                return result
            cooldown_manager.update(lesson_id)
        
        result["should_nudge"] = result["llm_result"].get("should_nudge", False)
        nudge = result["llm_result"].get("nudge")

        if result["should_nudge"] and nudge:
            result["nudge_text"] = nudge + " WHY: " + (result["llm_result"].get("why") or "")
            t = time.perf_counter()
            result["audio_generated"] = self.tts_service.speak(result["nudge_text"])
            times["tts"] = round(time.perf_counter() - t, 3)
            response_data["audio_generated"] = result["audio_generated"]
        
        times["total"] = round(time.perf_counter() - t_total, 3)
        print(f"  ⏱ speechbrain={times.get('speechbrain')}s | whisper={times.get('whisper')}s | similarity={times.get('similarity')}s | llm={times.get('llm')}s | tts={times.get('tts', '-')}s | TOTAL={times['total']}s")

        result["response_data"] = response_data
        return result
