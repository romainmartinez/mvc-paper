import pytest

from mvc.constants import RAW_DATA_DIR, MUSCLES
from mvc.fileio import from_matlab_to_pandas


@pytest.mark.parametrize("query", ["only_max", "100_points"])
@pytest.mark.parametrize("wide", [True, False])
def test_matlab_to_pandas(query, wide):
    d = from_matlab_to_pandas(RAW_DATA_DIR, query, MUSCLES, wide)

    if wide:
        computed_sum = (
            d.drop(["dataset", "participant", "muscle"], axis=1).sum().sum().round()
        )
        expected_shape = (1721, 19)
    else:
        computed_sum = d["mvc"].sum().round()
        expected_shape = (18465, 5)

    expected_sum = 1690 if query == "only_max" else 1970
    assert computed_sum == expected_sum
    assert d.shape == expected_shape
