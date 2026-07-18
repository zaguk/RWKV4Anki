from __future__ import annotations

# Inline CSS/JavaScript is intentionally kept readable as browser source.
# ruff: noqa: E501
from .setup_faq_html import render_setup_faq_overlay

# These fragments serve both the overlay launched from RWKV Settings and the
# settings-free first-run document assembled by config_html.py.  Keeping both
# presentations on the same HTML/CSS/JavaScript components avoids a second web
# build pipeline and preserves one visual and interaction contract.

SETUP_WIZARD_CSS = r"""
    .setup-launcher {
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px;
        align-items: center; margin-top: 18px; padding: 17px 19px;
        border: 1px solid rgba(75,123,236,.42); border-radius: 11px;
        background: linear-gradient(135deg, rgba(75,123,236,.13), rgba(75,123,236,.035));
    }
    .setup-launcher h2 { font-size: 17px; margin: 0 0 3px; }
    .setup-launcher p { color: var(--fg-subtle, #626872); margin: 0; max-width: 680px; }
    .setup-launcher .setup-kicker {
        color: var(--rwkv-accent); font-size: 10px; font-weight: 750;
        letter-spacing: .09em; margin-bottom: 4px; text-transform: uppercase;
    }
    .setup-launcher-button { font-weight: 700; padding: 9px 14px; }
    .setup-overlay {
        position: fixed; inset: 0; z-index: 100; display: grid; place-items: center;
        padding: 22px; background: rgba(18,22,28,.56); backdrop-filter: blur(2px);
    }
    .setup-overlay[hidden] { display: none; }
    .setup-dialog {
        display: flex; flex-direction: column; width: min(760px, 100%);
        max-height: min(760px, calc(100vh - 44px)); overflow: hidden;
        color: var(--rwkv-fg); background: var(--rwkv-canvas);
        border: 1px solid var(--rwkv-border); border-radius: 14px;
        box-shadow: 0 22px 70px rgba(0,0,0,.34);
    }
    .setup-dialog-header {
        display: flex; align-items: flex-start; justify-content: space-between;
        gap: 16px; padding: 18px 21px 14px; border-bottom: 1px solid var(--rwkv-border);
    }
    .setup-step-label {
        color: var(--rwkv-accent); font-size: 10px; font-weight: 750;
        letter-spacing: .09em; text-transform: uppercase;
    }
    .setup-dialog-title { font-size: 21px; line-height: 1.2; margin: 3px 0 0; }
    .setup-dialog-body { overflow: auto; padding: 20px 22px; }
    .setup-copy { color: var(--fg-subtle, #626872); margin: 0 0 13px; }
    .setup-help-hint {
        color: var(--fg-subtle, #626872); font-size: 11px; margin: -3px 0 13px;
    }
    .setup-callout {
        margin: 14px 0; padding: 11px 13px; border-left: 3px solid var(--rwkv-accent);
        border-radius: 4px; background: rgba(75,123,236,.09);
    }
    .setup-callout.warning { border-left-color: #d98b22; background: rgba(217,139,34,.1); }
    .setup-callout.success { border-left-color: #2c9b62; background: rgba(44,155,98,.1); }
    .setup-choice-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 17px; }
    .setup-choice {
        appearance: none; min-height: 92px; padding: 14px 15px; cursor: pointer;
        color: inherit; text-align: left; background: var(--frame-bg, rgba(127,127,127,.055));
        border: 1px solid var(--rwkv-border); border-radius: 9px;
    }
    .setup-choice:hover { border-color: rgba(75,123,236,.7); background: rgba(75,123,236,.08); }
    .setup-choice strong { display: block; font-size: 14px; margin-bottom: 4px; }
    .setup-choice span { color: var(--fg-subtle, #626872); font-size: 12px; }
    .setup-results { display: grid; gap: 9px; margin: 15px 0; }
    .setup-result-row {
        display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px;
        align-items: baseline; padding: 10px 12px; border: 1px solid var(--rwkv-border);
        border-radius: 8px; background: var(--frame-bg, rgba(127,127,127,.055));
    }
    .setup-result-row small { display: block; color: var(--fg-subtle, #626872); margin-top: 2px; }
    .setup-result-value { font-variant-numeric: tabular-nums; font-weight: 700; text-align: right; }
    .setup-selected { color: #237d50; }
    .setup-progress-wrap { margin: 18px 0 8px; }
    .setup-progress-track {
        height: 12px; overflow: hidden; border-radius: 99px; background: rgba(127,127,127,.22);
    }
    .setup-progress-bar {
        width: 0; height: 100%; border-radius: inherit; background: var(--rwkv-accent);
        transition: width .12s ease;
    }
    .setup-progress-track.indeterminate .setup-progress-bar {
        width: 36%; animation: setup-indeterminate 1.15s ease-in-out infinite;
    }
    @keyframes setup-indeterminate { from { transform: translateX(-110%); } to { transform: translateX(310%); } }
    .setup-progress-meta {
        display: flex; justify-content: space-between; gap: 12px; margin-top: 7px;
        color: var(--fg-subtle, #626872); font-size: 12px;
    }
    .setup-progress-label { margin-top: 8px; font-weight: 650; }
    .setup-change-list { margin: 12px 0 0; padding-left: 21px; }
    .setup-change-list li { margin: 5px 0; }
    .setup-dialog-footer {
        display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px;
        padding: 13px 20px 17px; border-top: 1px solid var(--rwkv-border);
    }
    .setup-dialog-footer .setup-back-button { margin-right: auto; }
    .setup-faq-launch {
        --setup-faq-button-bg: #6842b8;
        --setup-faq-button-hover: #58369f;
        display: flex; justify-content: flex-end; margin-top: 16px; padding-top: 10px;
        border-top: 1px solid var(--rwkv-border);
    }
    .setup-faq-launch > button.setup-faq-launch-button {
        background: var(--setup-faq-button-bg); border-color: var(--setup-faq-button-bg);
        border-radius: var(--rwkv-control-radius); box-shadow: none; color: #fff;
        margin: 0; padding-left: 13px; padding-right: 13px;
    }
    .setup-faq-launch > button.setup-faq-launch-button:hover:not(:disabled) {
        background: var(--setup-faq-button-hover);
        border-color: var(--setup-faq-button-hover); border-radius: var(--rwkv-control-radius);
        box-shadow: none;
    }
    .nightMode .setup-faq-launch, .night_mode .setup-faq-launch {
        --setup-faq-button-bg: #7956c7;
        --setup-faq-button-hover: #6846b4;
    }
    .setup-faq-overlay { z-index: 115; }
    .setup-faq-panel {
        display: flex; flex-direction: column; max-height: min(760px, calc(100vh - 40px));
        max-width: 760px; overflow: hidden; padding: 0; width: min(100%, 760px);
    }
    .setup-faq-header {
        align-items: flex-start; border-bottom: 1px solid var(--rwkv-border);
        display: flex; flex: 0 0 auto; gap: 16px; justify-content: space-between;
        padding: 18px 21px 20px;
    }
    .setup-faq-header .rwkv-overlay-title { margin: 3px 0 0; }
    .setup-faq-content { min-height: 0; overflow-y: auto; padding: 17px 22px 22px; }
    .setup-faq-list { border-top: 1px solid var(--rwkv-border); }
    .setup-faq-item { border-bottom: 1px solid var(--rwkv-border); }
    .setup-faq-disclosure {
        color: var(--rwkv-fg); font-size: 14px; font-weight: 680;
        padding: 13px 2px;
    }
    .setup-faq-answer {
        color: var(--fg-subtle, #626872); line-height: 1.5; padding: 0 4px 14px;
    }
    .setup-faq-answer-heading {
        color: var(--rwkv-fg); font-size: 13px; margin: 15px 0 6px;
    }
    .setup-faq-answer-heading:first-child { margin-top: 0; }
    .setup-faq-answer p { margin: 0 0 10px; }
    .setup-faq-answer p:last-child { margin-bottom: 0; }
    .setup-faq-answer ul, .setup-faq-answer ol { margin: 0 0 10px; padding-left: 22px; }
    .setup-faq-answer li + li { margin-top: 6px; }
    .setup-faq-answer .rwkv-notice { margin: 10px 0; }
    .setup-faq-answer .rwkv-notice:first-child { margin-top: 0; }
    .setup-faq-answer .rwkv-notice:last-child { margin-bottom: 0; }
    .setup-dialog button:disabled { cursor: not-allowed; opacity: .45; }
    .setup-night .setup-dialog, .nightMode .setup-dialog, .night_mode .setup-dialog { color: #e7e9ec; }
    .nightMode .setup-selected, .night_mode .setup-selected { color: #72d6a4; }
    @media (max-width: 620px) {
        .setup-launcher { grid-template-columns: 1fr; }
        .setup-launcher-button { justify-self: start; }
        .setup-overlay { padding: 8px; }
        .setup-dialog { max-height: calc(100vh - 16px); }
        .setup-choice-grid { grid-template-columns: 1fr; }
        .setup-result-row { grid-template-columns: 1fr; }
        .setup-result-value { text-align: left; }
    }
"""


def render_setup_launcher() -> str:
    return """
<section class="setup-launcher" aria-labelledby="rwkv-setup-heading">
    <div>
        <div class="setup-kicker">Guided setup</div>
        <h2 id="rwkv-setup-heading">Set up RWKV for this computer</h2>
        <p>Answer a few questions, test this computer with your collection, and review the suggested settings. Nothing is saved until you apply them.</p>
    </div>
    <button type="button" class="rwkv-button rwkv-button--primary setup-launcher-button"
            data-setup-event="open">Run Setup Wizard</button>
</section>
""".strip()


def render_setup_overlay(
    *,
    initially_open: bool = False,
    show_close_button: bool = True,
) -> str:
    hidden = "" if initially_open else " hidden"
    busy = ' aria-busy="true"' if initially_open else ""
    close_disabled = " disabled" if initially_open else ""
    close_button = (
        '<button type="button" class="setup-close rwkv-icon-close" data-setup-event="exit" '
        f'aria-label="Exit setup"{close_disabled}>&times;</button>'
        if show_close_button
        else ""
    )
    initial_body = (
        '<div class="setup-progress-wrap">'
        '<div class="setup-progress-label">Preparing Guided Setup…</div>'
        '<p class="setup-copy">Loading the first-time setup explanation.</p>'
        "</div>"
        if initially_open
        else ""
    )
    setup_overlay = f"""
<div class="setup-overlay" id="rwkv-setup-overlay" data-rwkv-keyboard-scope{hidden}>
    <section class="setup-dialog" role="dialog" aria-modal="true" aria-labelledby="rwkv-setup-title"{busy}>
        <header class="setup-dialog-header">
            <div>
                <div class="setup-step-label" id="rwkv-setup-step">Welcome</div>
                <h2 class="setup-dialog-title" id="rwkv-setup-title">Set up RWKV4Anki</h2>
            </div>
            {close_button}
        </header>
        <div class="setup-dialog-body" id="rwkv-setup-body">{initial_body}</div>
        <footer class="setup-dialog-footer" id="rwkv-setup-footer"></footer>
    </section>
</div>
""".strip()
    return setup_overlay + "\n" + render_setup_faq_overlay()


SETUP_WIZARD_SCRIPT = r"""
    const setupOverlay = document.getElementById('rwkv-setup-overlay');
    const setupBody = document.getElementById('rwkv-setup-body');
    const setupFooter = document.getElementById('rwkv-setup-footer');
    const setupTitle = document.getElementById('rwkv-setup-title');
    const setupStep = document.getElementById('rwkv-setup-step');
    const setupDialog = setupOverlay.querySelector('.setup-dialog');
    const setupClose = setupOverlay.querySelector('.setup-close');
    const setupFaq = document.getElementById('rwkv-setup-faq-overlay');
    const settingsShell = document.querySelector('.settings-shell');
    let setupOpen = false;
    function sendSetup(event, extra={}) {
        send('setup', Object.assign({event: event, values: values()}, extra));
    }
    document.addEventListener('click', (event) => {
        const button = event.target.closest('[data-setup-event]');
        if (!button || button.disabled) return;
        event.preventDefault();
        if (setupOpen) button.disabled = true;
        sendSetup(button.dataset.setupEvent, {value: button.dataset.setupValue || null});
    });
    document.addEventListener('click', (event) => {
        const open = event.target.closest('[data-setup-faq-open]');
        if (open) {
            event.preventDefault();
            window.RWKVModal?.showOverlay(setupFaq);
            return;
        }
        const close = event.target.closest('[data-setup-faq-close]');
        if (close) {
            event.preventDefault();
            window.RWKVModal?.hideOverlay(setupFaq);
        }
    });
    document.addEventListener('keydown', (event) => {
        if (event.defaultPrevented) return;
        if (event.key === 'Escape' && setupOpen) {
            event.preventDefault();
            sendSetup('exit');
            return;
        }
        if (event.key === 'Tab' && setupOpen) {
            const focusable = Array.from(setupDialog.querySelectorAll(
                'button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])'
            )).filter((element) => element.offsetParent !== null);
            if (!focusable.length) {
                event.preventDefault();
                return;
            }
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        }
    });
    window.rwkvSetupRender = (payload) => {
        setupOpen = true;
        setupOverlay.hidden = false;
        setupDialog.removeAttribute('aria-busy');
        setupStep.textContent = payload.step || 'Setup';
        setupTitle.textContent = payload.title || 'RWKV Setup';
        setupBody.innerHTML = payload.body || '';
        setupFooter.innerHTML = payload.footer || '';
        window.RWKVModal.initializeGlossaryTerms(setupBody);
        window.RWKVModal.initializeTooltips(setupBody);
        window.RWKVModal.initializeTooltips(setupFooter);
        if (setupClose) {
            const showCloseButton = payload.showCloseButton !== false
                && !Boolean(payload.busy);
            setupClose.hidden = !showCloseButton;
            setupClose.disabled = !showCloseButton;
        }
        const focusTarget = setupBody.querySelector('[data-setup-initial-focus]:not(:disabled)')
            || setupFooter.querySelector('[data-setup-initial-focus]:not(:disabled)')
            || setupBody.querySelector('button:not(:disabled), input:not(:disabled)')
            || setupFooter.querySelector('button:not(:disabled)');
        if (focusTarget) focusTarget.focus();
        if (settingsShell) {
            settingsShell.inert = true;
            settingsShell.setAttribute('aria-hidden', 'true');
        }
        if (window.rwkvSetSettingsFooterActive) {
            window.rwkvSetSettingsFooterActive(false);
        }
    };
    window.rwkvSetupProgress = (payload) => {
        const track = document.getElementById('rwkv-setup-progress-track');
        const bar = document.getElementById('rwkv-setup-progress-bar');
        const label = document.getElementById('rwkv-setup-progress-label');
        const count = document.getElementById('rwkv-setup-progress-count');
        const eta = document.getElementById('rwkv-setup-progress-eta');
        if (!track || !bar) return;
        const total = Math.max(0, Number(payload.total) || 0);
        const current = Math.max(0, Math.min(total, Number(payload.current) || 0));
        const determinate = total > 0;
        track.classList.toggle('indeterminate', !determinate);
        track.setAttribute('aria-valuemin', '0');
        track.setAttribute('aria-valuemax', String(determinate ? total : 1));
        track.setAttribute('aria-valuenow', String(determinate ? current : 0));
        if (determinate) bar.style.width = `${Math.round(current * 100 / total)}%`;
        if (label) label.textContent = payload.label || 'Working…';
        if (count) count.textContent = determinate ? `${current.toLocaleString()} / ${total.toLocaleString()}` : 'Working…';
        if (eta) eta.textContent = payload.eta || 'ETA unknown';
    };
    window.rwkvSetupClose = () => {
        if (setupFaq && !setupFaq.hidden) {
            window.RWKVModal?.hideOverlay(setupFaq, {restoreFocus: false});
        }
        setupOpen = false;
        setupOverlay.hidden = true;
        if (settingsShell) {
            settingsShell.inert = false;
            settingsShell.removeAttribute('aria-hidden');
        }
        if (window.rwkvSetSettingsFooterActive) {
            window.rwkvSetSettingsFooterActive(true);
        }
        setupBody.innerHTML = '';
        setupFooter.innerHTML = '';
        const launcher = document.querySelector('[data-setup-event="open"]');
        if (launcher) {
            launcher.disabled = false;
            launcher.focus();
        }
    };
"""
