"""Hash and verify passwords with Argon2id, never a weaker or custom scheme."""

# Import the high-level Argon2 password hasher.
from argon2 import PasswordHasher

# Import the Argon2 exceptions raised on hash mismatch or malformed hashes.
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# Import the algorithm-variant enum so Argon2id can be pinned explicitly.
from argon2.low_level import Type

# Enforce a length floor that matters far more than complexity theater (spec section 11).
MINIMUM_PASSWORD_LENGTH = 8

# Configure one process-wide hasher explicitly pinned to the Argon2id variant.
#
# argon2-cffi already defaults to Type.ID, but pin it explicitly so a future
# library default change can never silently downgrade the hashing scheme.
_password_hasher = PasswordHasher(type=Type.ID)


# Hash a plaintext password for storage; the plaintext itself is never stored.
def hash_password(plain_password: str) -> str:
    """Return an Argon2id hash string encoding the algorithm, params, and salt."""

    # Delegate salt generation and encoding entirely to the audited library.
    return _password_hasher.hash(plain_password)


# Verify a login attempt against a stored Argon2id hash.
def verify_password(plain_password: str, password_hash: str) -> bool:
    """Return True only if the password matches the stored Argon2id hash."""

    # Attempt verification and treat any failure mode as "does not match".
    try:
        # Compare using Argon2's constant-time verification routine.
        return _password_hasher.verify(password_hash, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        # Never raise on a wrong password; the caller only needs a boolean.
        return False


# Detect hashes that were created under outdated parameters.
def needs_rehash(password_hash: str) -> bool:
    """Return True when a stored hash should be re-hashed under current params."""

    # Ask the hasher whether stored parameters still match current settings.
    return _password_hasher.check_needs_rehash(password_hash)
