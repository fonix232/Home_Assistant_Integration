"""Test LAN-only tags: onboarded over mDNS and never seen over Bluetooth."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import CONNECTION_BLUETOOTH
from homeassistant.util import dt as dt_util
from opendisplay import OpenDisplayConnectionError
from PIL import Image as PILImage
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.opendisplay.ble_lock import async_get_ble_lock
from custom_components.opendisplay.const import (
    CONF_HOST,
    CONF_LAN_ONLY,
    CONF_PORT,
    CONF_TLS,
    DOMAIN,
    LAN_PROBE_INTERVAL,
)

from . import TEST_ADDRESS

pytestmark = pytest.mark.usefixtures("entity_registry_enabled_by_default")

CONNECTIVITY = "binary_sensor.opendisplay_1234_connectivity"
PROBE = "custom_components.opendisplay.coordinator.async_probe_tcp"


@pytest.fixture
def config_entry_data() -> dict[str, object]:
    """Return entry data for a tag onboarded over mDNS with no Bluetooth."""
    return {
        CONF_HOST: "192.168.1.9",
        CONF_PORT: 2446,
        CONF_TLS: False,
        CONF_LAN_ONLY: True,
    }


@pytest.fixture(autouse=True)
def mock_ble_device() -> Iterator[None]:
    """No Bluetooth device exists for the tag's MAC, anywhere."""
    with ExitStack() as stack:
        for module in ("", ".config_flow", ".transport", ".update"):
            stack.enter_context(
                patch(
                    f"custom_components.opendisplay{module}.async_ble_device_from_address",
                    return_value=None,
                )
            )
        yield


@pytest.fixture
def probe() -> Iterator[AsyncMock]:
    """Patch the TCP probe; it succeeds unless a test sets a side effect."""
    with patch(PROBE, new_callable=AsyncMock) as mock:
        yield mock


async def _tick(hass: HomeAssistant) -> None:
    """Advance past one probe interval."""
    async_fire_time_changed(hass, dt_util.utcnow() + LAN_PROBE_INTERVAL)
    await hass.async_block_till_done()


# -- setup --------------------------------------------------------------------


async def test_setup_interrogates_over_tcp(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_opendisplay_device_class: MagicMock,
    device_registry: dr.DeviceRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """Setup reads firmware and config over TCP, with no BLE device at all."""
    await setup_entry()

    assert mock_config_entry.state is ConfigEntryState.LOADED
    kwargs = mock_opendisplay_device_class.call_args.kwargs
    assert (kwargs["host"], kwargs["port"], kwargs["tls"]) == (
        "192.168.1.9",
        2446,
        False,
    )
    assert "mac_address" not in kwargs
    [device] = dr.async_entries_for_config_entry(
        device_registry, mock_config_entry.entry_id
    )
    assert (CONNECTION_BLUETOOTH, TEST_ADDRESS) in device.connections
    assert device.sw_version is not None


async def test_setup_retries_when_the_lan_endpoint_is_unreachable(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_opendisplay_device: MagicMock,
) -> None:
    """A failed TCP interrogation is a retryable setup failure."""
    mock_opendisplay_device.__aenter__.side_effect = OpenDisplayConnectionError(
        "refused"
    )
    mock_config_entry.add_to_hass(hass)

    await hass.config_entries.async_setup(mock_config_entry.entry_id)

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_only_lan_capable_entities_are_created(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    entity_registry: er.EntityRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """Advert-fed sensors, events and BLE OTA are not created for a LAN-only tag."""
    await setup_entry()

    entities = er.async_entries_for_config_entry(
        entity_registry, mock_config_entry.entry_id
    )
    assert {e.domain for e in entities} == {"binary_sensor", "image"}
    assert hass.states.get(CONNECTIVITY).state == STATE_ON


# -- availability -------------------------------------------------------------


async def test_connectivity_follows_the_tcp_probe(
    hass: HomeAssistant,
    probe: AsyncMock,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """The periodic probe, not BLE adverts, drives connectivity."""
    await setup_entry()

    probe.side_effect = OSError("host unreachable")
    await _tick(hass)
    assert hass.states.get(CONNECTIVITY).state == STATE_OFF

    probe.side_effect = None
    await _tick(hass)
    assert hass.states.get(CONNECTIVITY).state == STATE_ON
    probe.assert_awaited_with("192.168.1.9", 2446)


async def test_a_successful_probe_wakes_the_delivery_manager(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    probe: AsyncMock,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """A reachable tag is a wake, so queued work drains without an advert."""
    await setup_entry()
    manager = mock_config_entry.runtime_data.delivery

    with patch.object(manager, "notify_device_seen") as notify:
        await _tick(hass)

    notify.assert_called_once()


async def test_the_probe_stands_aside_while_a_session_holds_the_link(
    hass: HomeAssistant,
    probe: AsyncMock,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """The tag serves one LAN client; an open session already proves it is up."""
    await setup_entry()

    async with async_get_ble_lock(TEST_ADDRESS):
        await _tick(hass)

    probe.assert_not_awaited()
    assert hass.states.get(CONNECTIVITY).state == STATE_ON


# -- delivery -----------------------------------------------------------------


@pytest.fixture
def local_image(tmp_path: Path) -> Iterator[None]:
    """Resolve any media id to a small local PNG."""
    image_path = tmp_path / "test.png"
    PILImage.new("RGB", (10, 10)).save(image_path)
    media = MagicMock()
    media.path = image_path
    with patch(
        "custom_components.opendisplay.services.async_resolve_media",
        return_value=media,
    ):
        yield


async def _upload_image(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    device = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)[0]
    await hass.services.async_call(
        DOMAIN,
        "upload_image",
        {
            "device_id": device.id,
            "image": {
                "media_content_id": "media-source://local/test.png",
                "media_content_type": "image/png",
            },
        },
        blocking=True,
    )


@pytest.mark.usefixtures("local_image")
async def test_upload_goes_over_lan_without_a_recent_mdns_sighting(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_opendisplay_device_class: MagicMock,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """No mDNS freshness gate: a LAN-only tag is always delivered to over TCP."""
    await setup_entry()
    assert mock_config_entry.runtime_data.mdns_last_seen is None
    mock_opendisplay_device_class.reset_mock()

    await _upload_image(hass, mock_config_entry)

    kwargs = mock_opendisplay_device_class.call_args.kwargs
    assert kwargs["host"] == "192.168.1.9"
    assert "mac_address" not in kwargs
    mock_opendisplay_device_class.return_value.upload_prepared_image.assert_awaited_once()


@pytest.mark.usefixtures("local_image")
async def test_a_lan_failure_is_reported_without_a_ble_fallback(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_opendisplay_device_class: MagicMock,
    mock_opendisplay_device: MagicMock,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """A LAN failure surfaces as an upload error; BLE is never tried."""
    await setup_entry()
    mock_opendisplay_device_class.reset_mock()
    mock_opendisplay_device.__aenter__.side_effect = OpenDisplayConnectionError(
        "connection refused"
    )

    with pytest.raises(HomeAssistantError, match="connection refused"):
        await _upload_image(hass, mock_config_entry)

    assert mock_opendisplay_device_class.call_count == 1
    assert "host" in mock_opendisplay_device_class.call_args.kwargs
