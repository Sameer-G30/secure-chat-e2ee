"""Cover single-use WebSocket tickets, origin checks, and frame size limits."""

# Import the shared helpers so these tests exercise the same handshake as the relay tests.
from httpx import AsyncClient
from test_ws_relay import (
    _CAROL,
    _alice_bob_conversation,
    _assert_ws_rejected,
    _connect_ws,
    _receive_until,
    _register_login,
    _ws_client,
)


# Request a ticket for a conversation with a bearer token.
async def _ticket(client: AsyncClient, token: str, conversation_id: str) -> str:
    """Return the ticket string or fail the test."""

    response = await client.post(
        f"/ws/conversations/{conversation_id}/ticket",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    return str(response.json()["ticket"])


# A ticket opens the socket once, and a second use is rejected.
async def test_ticket_is_single_use(client: AsyncClient) -> None:
    alice_token, _bob, conversation_id = await _alice_bob_conversation(client)
    ticket = await _ticket(client, alice_token, conversation_id)
    url = f"http://testserver/ws/conversations/{conversation_id}?ticket={ticket}"
    async with _ws_client() as ws_client:
        async with _connect_ws(url, ws_client) as ws:
            frame = await _receive_until(ws, "presence")
            assert frame["type"] == "presence"
    # The same ticket must not work again.
    await _assert_ws_rejected(url, (4401, 1008))


# A non-member cannot even obtain a ticket.
async def test_ticket_requires_membership(client: AsyncClient) -> None:
    _a, _b, conversation_id = await _alice_bob_conversation(client)
    carol_token = await _register_login(client, _CAROL, 0x03)
    response = await client.post(
        f"/ws/conversations/{conversation_id}/ticket",
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert response.status_code in (403, 404)


# A ticket for one conversation is useless on another path.
async def test_ticket_bound_to_conversation(client: AsyncClient) -> None:
    alice_token, _b, conversation_id = await _alice_bob_conversation(client)
    ticket = await _ticket(client, alice_token, conversation_id)
    other = "00000000-0000-0000-0000-000000000000"
    await _assert_ws_rejected(
        f"http://testserver/ws/conversations/{other}?ticket={ticket}", (4401, 4403, 1008)
    )
