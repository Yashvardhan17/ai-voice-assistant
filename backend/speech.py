from functools import lru_cache
from pathlib import Path
import tempfile
from fastapi import UploadFile, HTTPException

@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel
    return WhisperModel('tiny',device='cpu',compute_type='int8')

async def transcribe(upload:UploadFile)->str:
    suffix=Path(upload.filename or 'audio.wav').suffix.lower()
    if suffix not in ('.wav','.mp3','.m4a','.ogg','.webm','.mp4'):
        raise HTTPException(415,'Unsupported audio file type')
    content=await upload.read(10*1024*1024+1)
    if not content:raise HTTPException(400,'Empty audio file')
    if len(content)>10*1024*1024:raise HTTPException(413,'Audio file exceeds 10 MB')
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
        f.write(content);path=f.name
    try:
        segments,_=whisper().transcribe(path,beam_size=2,language='en')
        return ' '.join(s.text.strip() for s in segments).strip()
    finally:Path(path).unlink(missing_ok=True)
