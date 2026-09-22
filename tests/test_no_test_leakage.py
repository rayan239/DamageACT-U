import pandas as pd
import pytest
from damageactu.utils.reproducibility import assert_no_test_rows


def test_rejects_test_rows():
    df = pd.DataFrame({"event_role":["train","test"]})
    with pytest.raises(RuntimeError):
        assert_no_test_rows(df)
