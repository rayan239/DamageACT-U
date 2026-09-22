from damageactu.data.event_split import TRAIN_EVENTS, TEST_EVENTS


def test_event_sets():
    assert len(TRAIN_EVENTS) == 13
    assert len(TEST_EVENTS) == 6
    assert set(TRAIN_EVENTS).isdisjoint(TEST_EVENTS)
