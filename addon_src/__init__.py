from __future__ import annotations

try:
    from .rwkvp_addon.constants import ADDON_NAME
    from .rwkvp_addon.vendor_bootstrap import ensure_vendor_paths
except ImportError:  # pytest imports this file as a top-level module
    from rwkvp_addon.constants import ADDON_NAME
    from rwkvp_addon.vendor_bootstrap import ensure_vendor_paths

ensure_vendor_paths()

try:
    try:
        from .rwkvp_addon.gui.menu import setup_menu
    except ImportError:
        from rwkvp_addon.gui.menu import setup_menu

    setup_menu()
except Exception as exc:  # pragma: no cover - only runs inside Anki at startup
    try:
        from aqt import mw

        try:
            from .rwkvp_addon.gui.web_message import show_web_warning
        except ImportError:
            from rwkvp_addon.gui.web_message import show_web_warning

        show_web_warning(
            f"{ADDON_NAME} could not finish loading.",
            details=str(exc),
            title=ADDON_NAME,
            parent=mw,
        )
    except ModuleNotFoundError:
        pass
    except Exception as display_error:
        raise exc from display_error
