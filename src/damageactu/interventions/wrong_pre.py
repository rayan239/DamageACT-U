from __future__ import annotations


def validate_donor_map(donor_df, buildings_df):
    by_id = buildings_df.copy()
    by_id["building_id"] = by_id["building_id"].astype(str)
    by_id = by_id.set_index("building_id", drop=False)


    required = {"target_building_id","donor_building_id"}
    missing = required - set(donor_df.columns)
    if missing:
        raise ValueError("Donor map missing columns: " + str(sorted(missing)))


    for r in donor_df.itertuples():
        tid = str(r.target_building_id)
        did = str(r.donor_building_id)
        if tid == did:
            raise RuntimeError("Target selected itself as donor.")
        t = by_id.loc[tid]
        d = by_id.loc[did]
        if str(t.scene_id) != str(d.scene_id):
            raise RuntimeError("Wrong-PRE donor crosses scene boundary.")
    return True
