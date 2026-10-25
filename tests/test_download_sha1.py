import hashlib

from src.data.download import sha1_file, verify_sha1


def test_sha1_and_verify(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    expected = hashlib.sha1(b"hello").hexdigest()
    assert sha1_file(p) == expected
    assert verify_sha1(p, expected)
    assert not verify_sha1(p, "0" * 40)
    assert verify_sha1(p, None)
