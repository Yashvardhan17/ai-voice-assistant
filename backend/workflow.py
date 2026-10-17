"""A bounded, deterministic LangGraph workflow with verified database tools."""
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from backend.database import account,outage_for_area,create_ticket
from backend.intent import predict
from backend.rag import retrieve

class State(TypedDict,total=False):
    text:str
    customer_id:int
    intent:str
    confidence:float
    answer:str
    steps:list[str]
    sources:list[dict]
    ticket:dict | None

def analyze(state:State)->dict:
    p=predict(state['text'])
    return {'intent':p['intent'],'confidence':p['confidence'],'steps':['Intent classified: '+p['intent']]}

def route(state:State)->str:
    if state['confidence']<0.22:return 'clarify'
    if state['intent'] in ('connectivity','service_outage','billing','account_access'):return 'tools'
    return 'knowledge'

def clarify(state:State)->dict:
    return {'answer':'Could you describe your issue in more detail so I can route it correctly?','steps':state['steps']+['Low-confidence clarification']}

def tools(state:State)->dict:
    acc=account(state['customer_id'])
    if not acc:return {'answer':'Customer not found. Select a valid demo account.','steps':state['steps']+['Customer lookup failed']}
    steps=state['steps']+['Verified synthetic customer '+str(acc['customer_id'])]
    intent=state['intent'];ticket=None
    if intent in ('connectivity','service_outage'):
        outage=outage_for_area(acc['area']);steps.append('Checked active outages for '+acc['area'])
        if outage:
            answer=f"There is an active outage in {acc['area']}: {outage['description']}. No ticket was created because a known outage exists."
        elif intent=='connectivity' and any(x in state['text'].lower() for x in ('ticket','complaint','report','create','raise','not working')):
            ticket=create_ticket(state['customer_id'],state['text']);steps.append('Created ticket #'+str(ticket['ticket_id']))
            answer=f"No outage is reported in {acc['area']}. Support ticket #{ticket['ticket_id']} was created for your issue."
        else:answer=f"No active outage is currently recorded in {acc['area']}. Would you like to create a support ticket?"
    elif intent=='billing':answer=f"Your synthetic account balance is INR {acc['balance_inr']}. This is demo data, not a real invoice."
    else:answer='For account access, use the password reset option. I cannot change credentials without verification.'
    return {'answer':answer,'steps':steps+['Verified tool outputs'],'ticket':ticket}

def knowledge(state:State)->dict:
    docs=retrieve(state['text']);steps=state['steps']+['Retrieved policy documents']
    if not docs:return {'answer':'I could not find a verified policy answer. Please contact support.','sources':[],'steps':steps}
    snippets=' '.join(x['text'] for x in docs[:2])
    return {'answer':f'According to our fictional knowledge base: {snippets}','sources':docs,'steps':steps+['Grounded answer in retrieved text']}

builder=StateGraph(State)
builder.add_node('analyze',analyze);builder.add_node('tools',tools);builder.add_node('knowledge',knowledge);builder.add_node('clarify',clarify)
builder.add_edge(START,'analyze');builder.add_conditional_edges('analyze',route,{'tools':'tools','knowledge':'knowledge','clarify':'clarify'})
for node in ('tools','knowledge','clarify'):builder.add_edge(node,END)
graph=builder.compile()

def execute(text:str,customer_id:int=1)->State:
    return graph.invoke({'text':text,'customer_id':customer_id,'steps':[],'sources':[],'ticket':None})
