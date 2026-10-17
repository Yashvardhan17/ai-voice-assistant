from backend.database import initialize,account,outage_for_area
from ml.train import SAMPLES

def test_data_has_six_classes():
    assert len(SAMPLES)==6
    assert all(len(examples)>=10 for examples in SAMPLES.values())

def test_seeded_customers_and_outage():
    initialize()
    assert account(1)['area']=='Central'
    assert account(2)['area']=='North'
    assert outage_for_area('North') is not None
    assert outage_for_area('Central') is None
