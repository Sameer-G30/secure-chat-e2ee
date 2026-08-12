"""Configure the shared slowapi limiter used by authentication endpoints."""

# Import slowapi's limiter class and the default per-client key function.
from slowapi import Limiter
from slowapi.util import get_remote_address

# Share one limiter instance across the app and every rate-limited router.
#
# Keying by remote address satisfies the spec's "login and registration
# endpoints must be rate-limited" requirement without needing an account
# identity, which is exactly what is available before authentication.
limiter = Limiter(key_func=get_remote_address)
