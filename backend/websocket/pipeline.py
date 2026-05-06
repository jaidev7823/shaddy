import time
import logging
import base64
from typing import Optional, Dict, Any
from backend.websocket.cooldown import cooldown_manager
from backend.services.vad_service import save_audio_separately

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, speaker_service, transcription_service, llm_service, tts_service):
        self.speaker_service = speaker_service
        self.transcription_service = transcription_service
        self.llm_service = llm_service
        self.tts_service = tts_service

    def _should_cancel(self, state) -> bool:
        """Helper to check and reset cancellation state."""
        if state and state.cancel_current:
            state.cancel_current = False
            return True
        return False

    async def process_utterance(self, full_audio: bytes, state=None) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        times = {}

        # Save audio file
        try:
            save_audio_separately(full_audio, base_folder="audio/saved_audio")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

        if state:
            state.processing = True

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

        try:
            # 1. Speaker Identification
            t = time.perf_counter()
            result["is_student"] = self.speaker_service.is_student_voice(full_audio)
            times["speechbrain"] = round(time.perf_counter() - t, 3)

            if result["is_student"]:
                print(f"  ⏱ speechbrain={times['speechbrain']}s — student voice, skipped")
                return result

            if self._should_cancel(state): return result

            # 2. Transcription
            t = time.perf_counter()
            result["transcript"] = self.transcription_service.transcribe_from_bytes(full_audio)
            times["whisper"] = round(time.perf_counter() - t, 3)

            if not result["transcript"]:
                print(f"  ⏱ speechbrain={times['speechbrain']}s | whisper={times['whisper']}s — no transcript")
                return result

            print(f'  heard: "{result["transcript"]}"')
            if self._should_cancel(state): return result

            # 3. Speaker Similarity
            t = time.perf_counter()
            result["similarity"] = self.speaker_service.get_speaker_similarity(full_audio)
            times["similarity"] = round(time.perf_counter() - t, 3)

            # 4. LLM Processing
            t = time.perf_counter()
            result["llm_result"] = self.llm_service.process_transcript(result["transcript"])
            times["llm"] = round(time.perf_counter() - t, 3)
            
            if self._should_cancel(state): return result

            # 5. Cooldown Logic
            llm_res = result["llm_result"]
            response_data = {
                "transcript": result["transcript"],
                "speaker_similarity": result["similarity"],
                "llm_response": llm_res,
            }

            lesson_id = llm_res.get("lesson_id")
            if lesson_id:
                if cooldown_manager.is_active(lesson_id):
                    result["cooldown_active"] = True
                    response_data["cooldown_active"] = True
                    result["response_data"] = response_data
                    print(f"  ⏱ Cooldown active for lesson {lesson_id}")
                    return result
                cooldown_manager.update(lesson_id)

            # 6. TTS Generation
            result["should_nudge"] = llm_res.get("should_nudge", False)
            nudge = llm_res.get("nudge")

            if result["should_nudge"] and nudge:
                result["nudge_text"] = f"{nudge}. say something like: {llm_res.get('sentence', '')}, because: {llm_res.get('why', '')}"
                # result["nudge_text"] = f"{nudge}." 
                
                t = time.perf_counter()
                audio_bytes = await self.tts_service.generate_audio(result["nudge_text"])
                times["tts"] = round(time.perf_counter() - t, 3)
                
                if audio_bytes:
                    result["audio_generated"] = True
                    # Save to output_path for the /audio/generated endpoint (backward compatibility)
                    with open(self.tts_service.output_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Encode audio bytes as base64 for WebSocket transmission
                    import base64
                    response_data["audio_data"] = base64.b64encode(audio_bytes).decode('utf-8')
                    response_data["audio_format"] = "wav"
                else:
                    result["audio_generated"] = False
                
                response_data["audio_generated"] = result["audio_generated"]

            # Finalize
            times["total"] = round(time.perf_counter() - t_total_start, 3)
            self._print_stats(times)
            
            result["response_data"] = response_data
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return result
        finally:
            if state:
                state.processing = False

    def _print_stats(self, times: dict):
        stats = " | ".join([f"{k}={v}s" for k, v in times.items() if k != "total"])
        print(f"  ⏱ {stats} | TOTAL={times.get('total')}s")
