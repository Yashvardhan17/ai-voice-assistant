import json
import os
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(page_title='Telecom AI Voice Assistant',page_icon='🎙️',layout='wide')
st.title('🎙️ Telecom AI Voice Assistant')
st.caption('Speech recognition • ML intent detection • LangGraph • SQL tools • RAG')
API=st.sidebar.text_input('Backend URL',os.getenv('BACKEND_URL','http://localhost:8000'))
customer=st.sidebar.selectbox('Synthetic customer',[1,2],format_func=lambda x:'Alex — Central' if x==1 else 'Sam — North')
with st.sidebar.expander('Example scenarios',expanded=True):
    st.markdown('**Alex (Central):** My internet is not working. Create a complaint.\n\n**Sam (North):** Check whether there is an outage in my area.\n\n**Billing:** What is my outstanding balance?\n\n**RAG:** What is the router restart policy?')
mode=st.radio('Input method',['Type a message','Record or upload audio'],horizontal=True)
message=''
if mode=='Type a message':
    message=st.text_input('Ask the assistant','My internet is not working. Please create a complaint.')
else:
    recording=st.audio_input('Record your question')
    upload=st.file_uploader('Or upload audio',type=['wav','mp3','m4a','ogg','webm','mp4'])
    audio=recording or upload
    if audio and st.button('Transcribe audio'):
        try:
            response=requests.post(API+'/transcribe',files={'file':(audio.name,audio.getvalue())},timeout=180)
            response.raise_for_status()
            st.session_state['transcript']=response.json()['text']
        except requests.RequestException as e:st.error(str(e))
    message=st.text_input('Transcribed text (editable)',value=st.session_state.get('transcript',''))
if st.button('Send to assistant',type='primary',disabled=not message.strip()):
    try:
        response=requests.post(API+'/chat',json={'text':message,'customer_id':customer},timeout=120)
        response.raise_for_status();st.session_state['last']=response.json()
    except requests.RequestException as e:st.error(f'Backend request failed: {e}')
if 'last' in st.session_state:
    r=st.session_state['last']
    st.subheader('Assistant answer');st.success(r.get('answer',''))
    a,b,c=st.columns(3)
    a.metric('Predicted intent',r.get('intent','—'))
    b.metric('Confidence',f"{r.get('confidence',0)*100:.1f}%")
    c.metric('Backend latency',f"{r.get('latency_ms',0):.0f} ms")
    with st.expander('Workflow trace',expanded=True):
        for item in r.get('steps',[]):st.write('✅',item)
    if r.get('ticket'):st.info('Ticket created: #'+str(r['ticket']['ticket_id']))
    if r.get('sources'):st.json(r['sources'])
    if st.button('Speak answer'):
        try:
            import pyttsx3
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                path=str(Path(d)/'answer.wav')
                engine=pyttsx3.init();engine.save_to_file(r['answer'],path);engine.runAndWait();engine.stop()
                if Path(path).exists() and Path(path).stat().st_size:st.audio(Path(path).read_bytes(),format='audio/wav')
                else:st.warning('Voice output unavailable. Text answer remains available.')
        except Exception as e:st.warning(f'TTS unavailable: {e}')
metrics=Path(__file__).resolve().parents[1]/'artifacts'/'metrics.json'
with st.expander('Model evaluation / Monitoring'):
    if metrics.exists():
        data=json.loads(metrics.read_text())
        st.json({name:{k:v for k,v in data[name].items() if k in ('accuracy','weighted_f1')} for name in ('logistic_regression','linear_svm')})
    else:st.write('Train the model to generate metrics.')
