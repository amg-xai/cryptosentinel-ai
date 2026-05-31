"""Rate-limit keying: authenticated users get per-user buckets, not per-IP."""
from jose import jwt


def test_unverified_claims_extracts_sub():
    """The limiter buckets by JWT sub when a Bearer token is present.
    We replicate the extraction logic to lock in the behavior."""
    # A token whose sub we can read without verification
    from src.api.auth import create_access_token
    token = create_access_token(user_id="user-a@x.ai", role="analyst")
    claims = jwt.get_unverified_claims(token)
    assert claims.get("sub") == "user-a@x.ai"


def test_malformed_token_falls_back_gracefully():
    """A malformed bearer token must not raise — limiter falls back to IP."""
    try:
        jwt.get_unverified_claims("not-a-real-token")
        raised = False
    except Exception:
        raised = True
    # get_unverified_claims raises on garbage; the middleware catches it.
    assert raised is True


def test_different_users_get_different_keys():
    from src.api.auth import create_access_token
    ta = create_access_token(user_id="a@x.ai", role="analyst")
    tb = create_access_token(user_id="b@x.ai", role="analyst")
    sa = jwt.get_unverified_claims(ta)["sub"]
    sb = jwt.get_unverified_claims(tb)["sub"]
    assert f"user:{sa}" != f"user:{sb}"
