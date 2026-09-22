from damageactu.interventions.donor_selection import DonorCandidate, choose_wrongpre_donor


def test_same_scene_and_deterministic():
    t = DonorCandidate("T","S",100.0,(0,0,10,10))
    a = DonorCandidate("A","S",100.0,(20,20,30,30))
    b = DonorCandidate("B","OTHER",100.0,(20,20,30,30))
    got = choose_wrongpre_donor(t,[b,a])
    assert got.building_id == "A"
    assert got.scene_id == "S"
