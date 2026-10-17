from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from backend.database import initialize, account
from backend.intent import predict

@asynccontextmanager
async def lifespan(app:FastAPI):
    initialize()
    yield
app=FastAPI(title='Telecom Voice Assistant',version='0.1.0',lifespan=lifespan)
class ChatRequest(BaseModel):
    text:str=Field(min_length=2,max_length=2000)
    customer_id:int=Field(default=1,ge=1)
class IntentRequest(BaseModel):
    text:str=Field(min_length=2)
@app.get('/health')
def health():return {'status':'ok','service':'telecom-ai'}
@app.post('/intent')
def intent(req:IntentRequest):return predict(req.text)
@app.get('/customer/{customer_id}')
def customer(customer_id:int):
    result=account(customer_id)
    if not result:raise HTTPException(404,'Customer not found')
    return result
@app.post('/chat')
def chat(req:ChatRequest):
    from backend.workflow import execute
    start=perf_counter()
    try:result=execute(req.text,req.customer_id)
    except Exception as e:raise HTTPException(500,detail=str(e)) from e
    return {**result,'latency_ms':round((perf_counter()-start)*1000,2)}
@app.post('/transcribe')
async def transcription(file:UploadFile=File(...)):
    from backend.speech import transcribe
    start=perf_counter();text=await transcribe(file)
    return {'text':text,'latency_ms':round((perf_counter()-start)*1000,2)}
