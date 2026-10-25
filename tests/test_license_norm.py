import pytest

from src.data.license_norm import license_family


@pytest.mark.parametrize("raw,expected", [
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "by-nc-sa"),
    ("https://creativecommons.org/licenses/by/4.0/", "by"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("Free Art License", "unknown"),
    ("", "unknown"),
    (None, "unknown"),
    (float("nan"), "unknown"),
])
def test_license_family(raw, expected):
    assert license_family(raw) == expected
