from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT=Path(__file__).resolve().parents[1]
engine=create_engine(f'sqlite:///{ROOT / "data" / "telecom.db"}',connect_args={'check_same_thread':False})
SessionLocal=sessionmaker(bind=engine)
class Base(DeclarativeBase): pass
class Customer(Base):
    __tablename__='customers'
    id=Column(Integer,primary_key=True)
    name=Column(String,nullable=False)
    area=Column(String,nullable=False)
    plan=Column(String,nullable=False)
    balance=Column(Integer,nullable=False,default=0)
class Outage(Base):
    __tablename__='outages'
    id=Column(Integer,primary_key=True)
    area=Column(String,nullable=False)
    active=Column(Boolean,default=True)
    description=Column(String,nullable=False)
class Ticket(Base):
    __tablename__='tickets'
    id=Column(Integer,primary_key=True)
    customer_id=Column(Integer,ForeignKey('customers.id'),nullable=False)
    issue=Column(String,nullable=False)
    created_at=Column(DateTime,default=lambda:datetime.now(timezone.utc))

def initialize():
    (ROOT/'data').mkdir(exist_ok=True)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.get(Customer,1) is None:
            db.add_all([Customer(id=1,name='Alex Demo',area='Central',plan='Fiber 100',balance=899),Customer(id=2,name='Sam Demo',area='North',plan='Fiber 300',balance=1299)])
        if not db.execute(select(Outage)).first():
            db.add(Outage(area='North',active=True,description='Fiber maintenance affecting North area'))
        db.commit()

def account(customer_id:int):
    with SessionLocal() as db:
        c=db.get(Customer,customer_id)
        return None if not c else {'customer_id':c.id,'name':c.name,'area':c.area,'plan':c.plan,'balance_inr':c.balance}

def outage_for_area(area:str):
    with SessionLocal() as db:
        found=db.execute(select(Outage).where(Outage.area==area,Outage.active.is_(True))).scalars().first()
        return None if not found else {'outage_id':found.id,'area':found.area,'description':found.description}

def create_ticket(customer_id:int,issue:str):
    with SessionLocal() as db:
        if not db.get(Customer,customer_id): raise ValueError('Customer not found')
        ticket=Ticket(customer_id=customer_id,issue=issue)
        db.add(ticket);db.commit();db.refresh(ticket)
        return {'ticket_id':ticket.id,'customer_id':customer_id,'issue':issue}
