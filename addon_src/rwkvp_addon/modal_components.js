(() => {
    "use strict";

    const bootstrap = window.RWKV_MODAL_BOOTSTRAP || {};
    const rootSelector = String(
        bootstrap.rootSelector || ".rwkv-modal-shell[data-rwkv-generation]"
    );
    const root = document.querySelector(rootSelector);
    if (!root) {
        return;
    }

    let requestSequence = 0;
    let activeOverlay = null;
    let focusBeforeOverlay = null;
    let activeProgressToken = null;
    let activeMessageToken = null;
    let activePromptState = null;
    let activeTooltipTarget = null;
    let tooltipPopover = null;
    let tooltipShowFrame = null;
    let tooltipHoverTimer = null;
    let tooltipHoverTarget = null;
    let tooltipGlobalListenersInitialized = false;
    let activePopupControl = null;
    let popupControlSequence = 0;
    let popupControlObserver = null;
    const popupControlRecords = new Map();
    const tooltipHoverDelayMs = 500;
    const popupControlMode = String(
        bootstrap.popupControlMode
        || window.RWKV_MODAL_POPUP_CONTROL_MODE
        || "auto"
    );
    const windowsUserAgent = [
        window.navigator?.userAgent,
        window.navigator?.platform,
        window.navigator?.userAgentData?.platform,
    ].filter(Boolean).join(" ");
    const useInPagePopupControls = popupControlMode === "in-page"
        || (popupControlMode !== "native" && /Windows/i.test(windowsUserAgent));

    const parsePayload = (element) => {
        const encoded = element?.dataset?.rwkvPayload;
        if (!encoded) {
            return {};
        }
        try {
            const parsed = JSON.parse(encoded);
            return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
        } catch (_error) {
            return {};
        }
    };

    const announce = (message) => {
        const region = document.getElementById("rwkv-dialog-announcer");
        if (region) {
            region.textContent = "";
            window.requestAnimationFrame(() => { region.textContent = String(message || ""); });
        }
    };

    const send = (action, payload = {}) => new Promise((resolve, reject) => {
        if (typeof window.pycmd !== "function") {
            reject(new Error("The Anki bridge is not ready."));
            return;
        }
        let effectivePayload = payload;
        const payloadProvider = window.RWKV_MODAL_PAYLOAD_PROVIDER;
        if (typeof payloadProvider === "function") {
            try {
                const provided = payloadProvider(String(action), payload);
                if (provided && typeof provided === "object" && !Array.isArray(provided)) {
                    effectivePayload = provided;
                }
            } catch (error) {
                reject(error);
                return;
            }
        }
        const requestId = `${Date.now()}-${++requestSequence}`;
        const command = bootstrap.bridgeMode === "flat"
            ? { ...effectivePayload, action: String(action) }
            : {
                generation: Number(bootstrap.generation),
                action: String(action),
                payload: effectivePayload,
                requestId,
            };
        const encoded = String(bootstrap.bridgePrefix) + JSON.stringify(command);
        if (bootstrap.bridgeExpectsReply === false) {
            try {
                window.pycmd(encoded);
                resolve(undefined);
            } catch (error) {
                reject(error);
            }
            return;
        }
        window.pycmd(encoded, (reply) => {
            if (!reply || reply.ok !== true) {
                const error = new Error(reply?.error?.message || "The dialog command failed.");
                error.code = reply?.error?.code || "bridge-error";
                announce(error.message);
                root.dispatchEvent(new CustomEvent("rwkv:bridge-error", { detail: reply }));
                reject(error);
                return;
            }
            resolve(reply.result);
        });
    });

    const focusableElements = (container) => Array.from(container.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
        'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )).filter((element) => !element.hidden && element.getAttribute("aria-hidden") !== "true");

    const glossaryTerms = (container = root) => Array.from(
        container.querySelectorAll(".glossary-term")
    );

    const positionGlossary = (term) => {
        const popover = term?.querySelector(".glossary-popover");
        if (!popover) {
            return;
        }
        const termRect = term.getBoundingClientRect();
        const popoverRect = popover.getBoundingClientRect();
        const margin = 12;
        const left = Math.max(
            margin,
            Math.min(termRect.left, window.innerWidth - popoverRect.width - margin),
        );
        let top = termRect.bottom + 8;
        if (
            top + popoverRect.height > window.innerHeight - margin
            && termRect.top > popoverRect.height + margin + 8
        ) {
            top = termRect.top - popoverRect.height - 8;
        }
        popover.style.left = `${Math.round(left)}px`;
        popover.style.top = `${Math.round(Math.max(margin, top))}px`;
    };

    const positionVisibleGlossaryTerms = () => {
        for (const term of glossaryTerms()) {
            if (term.matches(":hover, :focus") || term.classList.contains("is-open")) {
                positionGlossary(term);
            }
        }
    };

    const closeGlossaryTerms = (except = null) => {
        for (const term of glossaryTerms()) {
            if (term === except) {
                continue;
            }
            term.classList.remove("is-open");
            term.setAttribute("aria-expanded", "false");
        }
    };

    const initializeGlossaryTerms = (container = root) => {
        for (const term of glossaryTerms(container)) {
            if (term.hasAttribute("data-rwkv-glossary-bound")) {
                continue;
            }
            term.setAttribute("data-rwkv-glossary-bound", "true");
            term.addEventListener("mouseenter", () => {
                term.classList.remove("tooltip-dismissed");
                positionGlossary(term);
            });
            term.addEventListener("focus", () => positionGlossary(term));
            term.addEventListener("blur", () => term.classList.remove("tooltip-dismissed"));
            term.addEventListener("click", () => {
                const opening = !term.classList.contains("is-open");
                closeGlossaryTerms(term);
                term.classList.toggle("is-open", opening);
                term.classList.toggle("tooltip-dismissed", !opening);
                term.setAttribute("aria-expanded", String(opening));
                positionGlossary(term);
            });
            term.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    term.click();
                } else if (event.key === "Escape") {
                    event.preventDefault();
                    term.classList.remove("is-open");
                    term.classList.add("tooltip-dismissed");
                    term.setAttribute("aria-expanded", "false");
                }
            });
        }
        window.addEventListener("resize", positionVisibleGlossaryTerms);
        window.addEventListener("scroll", positionVisibleGlossaryTerms, true);
    };

    const positionTooltip = (target) => {
        if (!target || !tooltipPopover || tooltipPopover.hidden) {
            return;
        }
        const targetRect = target.getBoundingClientRect();
        const popoverRect = tooltipPopover.getBoundingClientRect();
        const margin = 12;
        const left = Math.max(
            margin,
            Math.min(targetRect.left, window.innerWidth - popoverRect.width - margin),
        );
        let top = targetRect.bottom + 8;
        if (
            top + popoverRect.height > window.innerHeight - margin
            && targetRect.top > popoverRect.height + margin + 8
        ) {
            top = targetRect.top - popoverRect.height - 8;
        }
        tooltipPopover.style.left = `${Math.round(left)}px`;
        tooltipPopover.style.top = `${Math.round(Math.max(margin, top))}px`;
    };

    const cancelTooltipHover = (target = null) => {
        if (target && target !== tooltipHoverTarget) {
            return;
        }
        if (tooltipHoverTimer !== null) {
            window.clearTimeout(tooltipHoverTimer);
            tooltipHoverTimer = null;
        }
        tooltipHoverTarget = null;
    };

    const scheduleTooltipForHover = (target) => {
        cancelTooltipHover();
        tooltipHoverTarget = target;
        tooltipHoverTimer = window.setTimeout(() => {
            tooltipHoverTimer = null;
            tooltipHoverTarget = null;
            if (target.matches(":hover")) {
                showTooltip(target);
            }
        }, tooltipHoverDelayMs);
    };

    const hideTooltip = (target = null, { dismiss = false } = {}) => {
        cancelTooltipHover(target);
        if (target && target !== activeTooltipTarget) {
            return;
        }
        if (dismiss && activeTooltipTarget) {
            activeTooltipTarget.classList.add("rwkv-tooltip-dismissed");
        }
        activeTooltipTarget = null;
        if (tooltipShowFrame !== null) {
            window.cancelAnimationFrame(tooltipShowFrame);
            tooltipShowFrame = null;
        }
        if (tooltipPopover) {
            const wasVisible = tooltipPopover.hasAttribute("data-visible");
            tooltipPopover.removeAttribute("data-visible");
            tooltipPopover.setAttribute("aria-hidden", "true");
            if (!wasVisible) {
                tooltipPopover.hidden = true;
            }
        }
    };

    const showTooltip = (target) => {
        cancelTooltipHover();
        const message = String(target?.dataset?.rwkvTooltip || "").trim();
        if (!message || target.classList.contains("rwkv-tooltip-dismissed")) {
            return;
        }
        if (tooltipShowFrame !== null) {
            window.cancelAnimationFrame(tooltipShowFrame);
            tooltipShowFrame = null;
        }
        activeTooltipTarget = target;
        tooltipPopover.textContent = message;
        tooltipPopover.hidden = false;
        tooltipPopover.setAttribute("aria-hidden", "false");
        positionTooltip(target);
        if (!tooltipPopover.hasAttribute("data-visible")) {
            tooltipShowFrame = window.requestAnimationFrame(() => {
                tooltipShowFrame = null;
                if (activeTooltipTarget === target) {
                    tooltipPopover.setAttribute("data-visible", "true");
                }
            });
        }
    };

    const initializeTooltips = (container = root) => {
        const targets = Array.from(container.querySelectorAll("[data-rwkv-tooltip]"));
        if (!targets.length) {
            return;
        }
        if (!tooltipPopover) {
            tooltipPopover = document.createElement("div");
            tooltipPopover.className = "rwkv-tooltip-popover";
            tooltipPopover.id = "rwkv-shared-tooltip";
            tooltipPopover.setAttribute("role", "tooltip");
            tooltipPopover.setAttribute("aria-hidden", "true");
            tooltipPopover.hidden = true;
            tooltipPopover.addEventListener("transitionend", (event) => {
                if (
                    event.propertyName === "opacity"
                    && !tooltipPopover.hasAttribute("data-visible")
                ) {
                    tooltipPopover.hidden = true;
                }
            });
            document.body.appendChild(tooltipPopover);
        }
        for (const target of targets) {
            if (target.hasAttribute("data-rwkv-tooltip-bound")) {
                continue;
            }
            target.setAttribute("data-rwkv-tooltip-bound", "true");
            target.addEventListener("mouseenter", () => {
                target.classList.remove("rwkv-tooltip-dismissed");
                scheduleTooltipForHover(target);
            });
            target.addEventListener("mouseleave", () => {
                target.classList.remove("rwkv-tooltip-dismissed");
                if (!target.matches(":focus")) {
                    hideTooltip(target);
                }
            });
            target.addEventListener("focus", () => showTooltip(target));
            target.addEventListener("blur", () => {
                target.classList.remove("rwkv-tooltip-dismissed");
                if (!target.matches(":hover")) {
                    hideTooltip(target);
                }
            });
        }
        if (!tooltipGlobalListenersInitialized) {
            tooltipGlobalListenersInitialized = true;
            window.addEventListener("resize", () => positionTooltip(activeTooltipTarget));
            window.addEventListener("scroll", () => positionTooltip(activeTooltipTarget), true);
        }
    };

    const sortableValue = (row, key, kind) => {
        let values = {};
        try {
            values = JSON.parse(row.dataset.rwkvSortValues || "{}");
        } catch (_error) {
            return null;
        }
        const raw = values?.[key];
        if (raw === null || raw === undefined || raw === "") {
            return null;
        }
        if (kind === "number") {
            const numeric = Number(raw);
            return Number.isFinite(numeric) ? numeric : null;
        }
        return String(raw);
    };

    const compareSortableValues = (left, right, kind, direction) => {
        const leftMissing = left === null;
        const rightMissing = right === null;
        if (leftMissing || rightMissing) {
            if (leftMissing && rightMissing) return 0;
            return leftMissing ? 1 : -1;
        }
        const compared = kind === "number"
            ? left - right
            : left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" });
        return direction === "descending" ? -compared : compared;
    };

    const initializeSortableTables = (container = root) => {
        for (const table of container.querySelectorAll("[data-rwkv-sortable-table]")) {
            const body = table.querySelector("tbody");
            if (!body) {
                continue;
            }
            const originalRows = Array.from(body.querySelectorAll("tr[data-rwkv-sort-values]"));
            originalRows.forEach((row, index) => {
                if (!row.hasAttribute("data-rwkv-sort-original-index")) {
                    row.dataset.rwkvSortOriginalIndex = String(index);
                }
            });
            for (const button of table.querySelectorAll("[data-rwkv-sort-key]")) {
                if (button.hasAttribute("data-rwkv-sort-bound")) {
                    continue;
                }
                button.setAttribute("data-rwkv-sort-bound", "true");
                button.addEventListener("click", (event) => {
                    event.preventDefault();
                    const key = String(button.dataset.rwkvSortKey || "");
                    const kind = button.dataset.rwkvSortKind === "number" ? "number" : "text";
                    const label = String(button.dataset.rwkvSortLabel || key || "column");
                    const direction = (
                        table.dataset.rwkvSortKey === key
                        && table.dataset.rwkvSortDirection === "ascending"
                    ) ? "descending" : "ascending";
                    const rows = Array.from(
                        body.querySelectorAll("tr[data-rwkv-sort-values]")
                    ).map((row) => ({
                        row,
                        value: sortableValue(row, key, kind),
                        originalIndex: Number(row.dataset.rwkvSortOriginalIndex || 0),
                    }));
                    rows.sort((left, right) => {
                        const compared = compareSortableValues(
                            left.value,
                            right.value,
                            kind,
                            direction,
                        );
                        return compared || left.originalIndex - right.originalIndex;
                    });
                    for (const entry of rows) {
                        body.appendChild(entry.row);
                    }

                    const sortButtons = Array.from(
                        table.querySelectorAll("[data-rwkv-sort-key]")
                    );
                    for (const candidate of sortButtons) {
                        candidate.setAttribute("aria-pressed", "false");
                        candidate.removeAttribute("data-rwkv-sort-direction");
                        const candidateLabel = String(
                            candidate.dataset.rwkvSortLabel
                            || candidate.dataset.rwkvSortKey
                            || "column"
                        );
                        candidate.setAttribute(
                            "aria-label",
                            `Sort by ${candidateLabel}, ascending`,
                        );
                        const indicator = candidate.querySelector(
                            ".rwkv-table-sort-indicator"
                        );
                        if (indicator) indicator.textContent = "";
                        candidate.closest("th")?.removeAttribute("aria-sort");
                    }
                    table.dataset.rwkvSortKey = key;
                    table.dataset.rwkvSortDirection = direction;
                    button.setAttribute("aria-pressed", "true");
                    button.setAttribute("data-rwkv-sort-direction", direction);
                    button.setAttribute(
                        "aria-label",
                        `Sort by ${label}, ${direction === "ascending" ? "descending" : "ascending"}`,
                    );
                    button.closest("th")?.setAttribute("aria-sort", direction);
                    const indicator = button.querySelector(".rwkv-table-sort-indicator");
                    if (indicator) {
                        indicator.textContent = direction === "ascending" ? "▲" : "▼";
                    }
                    announce(`Sorted by ${label}, ${direction}.`);
                });
            }
        }
    };

    /*
     * Chromium renders select and temporal-input pickers in separate native
     * popup widgets.  Qt WebEngine can repeatedly enlarge those widgets on
     * some Windows installations, leaving a growing white tail.  On
     * Windows we keep popup controls inside this document instead. Dates use
     * the shared text entry/calendar on every platform: native segmented date
     * editors emit change while typing, which can rerender a workflow before
     * the user finishes the year. Original form controls remain authoritative
     * for Python bridges, local scripts, and FormData.
     */
    const popupCandidates = (container, selector) => {
        const candidates = [];
        if (container?.matches?.(selector)) {
            candidates.push(container);
        }
        if (container?.querySelectorAll) {
            candidates.push(...container.querySelectorAll(selector));
        }
        return candidates;
    };

    const dispatchPopupSourceEvent = (source, type) => {
        source.dispatchEvent(new CustomEvent(type, { bubbles: true }));
    };

    const popupOwnerId = (element) => String(
        element?.closest?.("[data-rwkv-popup-owner]")?.dataset?.rwkvPopupOwner || ""
    );

    const closePopupControl = ({ restoreFocus = false } = {}) => {
        const record = activePopupControl;
        if (!record) {
            return false;
        }
        record.panel.hidden = true;
        record.panel.setAttribute("aria-hidden", "true");
        record.trigger.setAttribute("aria-expanded", "false");
        activePopupControl = null;
        if (restoreFocus && record.trigger.isConnected) {
            record.trigger.focus();
        }
        return true;
    };

    const positionPopupControl = (record = activePopupControl) => {
        if (!record || record.panel.hidden || !record.trigger.isConnected) {
            return;
        }
        const margin = 12;
        const gap = 6;
        const triggerRect = record.trigger.getBoundingClientRect();
        const availableWidth = Math.max(80, window.innerWidth - margin * 2);
        record.panel.style.maxHeight = "";
        record.panel.style.maxWidth = `${Math.floor(availableWidth)}px`;
        record.panel.style.minWidth = `${Math.min(
            Math.max(0, triggerRect.width),
            availableWidth,
        )}px`;
        record.panel.style.visibility = "hidden";
        const panelRect = record.panel.getBoundingClientRect();
        const panelWidth = Math.min(
            Math.max(triggerRect.width, panelRect.width || 0),
            availableWidth,
        );
        const left = Math.max(
            margin,
            Math.min(triggerRect.left, window.innerWidth - panelWidth - margin),
        );
        const spaceBelow = window.innerHeight - triggerRect.bottom - margin - gap;
        const spaceAbove = triggerRect.top - margin - gap;
        const panelHeight = Math.min(panelRect.height || 0, 320);
        const openAbove = panelHeight > spaceBelow && spaceAbove > spaceBelow;
        record.panel.style.maxHeight = `${Math.max(
            64,
            Math.floor(openAbove ? spaceAbove : spaceBelow),
        )}px`;
        const top = openAbove
            ? Math.max(margin, triggerRect.top - panelHeight - gap)
            : Math.min(window.innerHeight - margin, triggerRect.bottom + gap);
        record.panel.style.left = `${Math.round(left)}px`;
        record.panel.style.top = `${Math.round(top)}px`;
        record.panel.style.visibility = "visible";
    };

    const openPopupControl = (record, { focusSelected = false } = {}) => {
        if (record.trigger.disabled || record.source.disabled) {
            return false;
        }
        if (activePopupControl && activePopupControl !== record) {
            closePopupControl();
        } else if (activePopupControl === record) {
            return true;
        }
        if (typeof record.renderPanel === "function") {
            record.renderPanel();
        }
        record.panel.hidden = false;
        record.panel.setAttribute("aria-hidden", "false");
        record.trigger.setAttribute("aria-expanded", "true");
        activePopupControl = record;
        positionPopupControl(record);
        window.requestAnimationFrame(() => {
            if (activePopupControl !== record) {
                return;
            }
            positionPopupControl(record);
            if (focusSelected) {
                const selected = record.panel.querySelector(
                    '[role="option"][aria-selected="true"]:not([disabled])'
                );
                (selected || record.panel.querySelector('[role="option"]:not([disabled])'))
                    ?.focus();
            }
        });
        return true;
    };

    const popupAccessibleLabel = (source, fallback) => {
        if (source.getAttribute("aria-label")) {
            return source.getAttribute("aria-label");
        }
        if (source.id) {
            const explicit = Array.from(document.querySelectorAll("label[for]"))
                .find((label) => label.getAttribute("for") === source.id);
            if (explicit?.textContent?.trim()) {
                return explicit.textContent.trim();
            }
        }
        const fieldLabel = source.closest(".rwkv-field")
            ?.querySelector(".rwkv-field__label, .rwkv-checkbox__label, .rwkv-switch__label");
        const partLabel = source.closest("label")?.querySelector(".datetime-part-label");
        return fieldLabel?.textContent?.trim()
            || partLabel?.textContent?.trim()
            || String(fallback);
    };

    const transferInitialFocus = (source, target) => {
        if (!source.hasAttribute("data-rwkv-initial-focus")) {
            return;
        }
        source.removeAttribute("data-rwkv-initial-focus");
        target.setAttribute("data-rwkv-initial-focus", "");
    };

    const rewireExplicitLabels = (source, target) => {
        if (!target.id) {
            return;
        }
        if (source.id) {
            for (const label of document.querySelectorAll("label[for]")) {
                if (label.getAttribute("for") === source.id) {
                    label.setAttribute("for", target.id);
                }
            }
        }
        const wrappingLabel = source.closest("label");
        const labelledId = wrappingLabel?.getAttribute("for");
        if (wrappingLabel && (!labelledId || labelledId === source.id)) {
            wrappingLabel.setAttribute("for", target.id);
        }
    };

    const createPopupPanel = (record, className, role, label) => {
        const panel = document.createElement("div");
        panel.className = className;
        panel.id = `${record.id}-panel`;
        panel.dataset.rwkvPopupOwner = record.id;
        panel.setAttribute("role", role);
        panel.setAttribute("aria-label", label);
        panel.setAttribute("aria-hidden", "true");
        panel.hidden = true;
        // Keep the portal under the invoking keyboard/modal scope so theme
        // variables and focus containment remain identical to the source.
        // Fixed positioning still keeps it out of card/table overflow geometry.
        const portalOwner = record.source.closest(
            "[data-rwkv-overlay], [data-rwkv-keyboard-scope]"
        ) || root;
        portalOwner.appendChild(panel);
        return panel;
    };

    const sourceOptions = (source) => Array.from(source.querySelectorAll("option"));

    const selectedSourceOption = (source, options = sourceOptions(source)) => {
        return options.find((option) => String(option.value) === String(source.value))
            || options.find((option) => option.selected || option.hasAttribute("selected"))
            || options.find((option) => !option.disabled)
            || options[0]
            || null;
    };

    const enhanceSelectControl = (source) => {
        if (
            popupControlRecords.has(source)
            || source.hasAttribute("multiple")
            || Number(source.getAttribute("size") || 0) > 1
            || source.hasAttribute("data-rwkv-native-popup")
        ) {
            return;
        }
        const id = `rwkv-popup-${++popupControlSequence}`;
        const parent = source.parentElement;
        if (!parent) {
            return;
        }
        const wrapper = document.createElement("div");
        wrapper.className = "rwkv-popup-select";
        wrapper.dataset.rwkvPopupOwner = id;
        parent.insertBefore(wrapper, source);
        wrapper.appendChild(source);
        source.classList.add("rwkv-popup-source");
        source.setAttribute("aria-hidden", "true");
        source.tabIndex = -1;

        const trigger = document.createElement("button");
        trigger.className = "rwkv-popup-select__trigger";
        trigger.type = "button";
        trigger.id = `${id}-trigger`;
        trigger.setAttribute("role", "combobox");
        trigger.setAttribute("aria-haspopup", "listbox");
        trigger.setAttribute("aria-expanded", "false");
        trigger.setAttribute("aria-label", popupAccessibleLabel(source, "Choose an option"));
        const describedBy = source.getAttribute("aria-describedby");
        if (describedBy) {
            trigger.setAttribute("aria-describedby", describedBy);
        }
        const valueLabel = document.createElement("span");
        valueLabel.className = "rwkv-popup-select__value";
        const indicator = document.createElement("span");
        indicator.className = "rwkv-popup-select__indicator";
        indicator.setAttribute("aria-hidden", "true");
        indicator.textContent = "▾";
        trigger.appendChild(valueLabel);
        trigger.appendChild(indicator);
        wrapper.appendChild(trigger);
        transferInitialFocus(source, trigger);
        rewireExplicitLabels(source, trigger);

        const record = {
            id,
            kind: "select",
            source,
            wrapper,
            trigger,
            panel: null,
            refresh: null,
        };
        const panel = createPopupPanel(
            record,
            "rwkv-popup-listbox",
            "listbox",
            trigger.getAttribute("aria-label") || "Options",
        );
        record.panel = panel;
        trigger.setAttribute("aria-controls", panel.id);

        let renderedOptions = [];
        let optionControls = [];
        const refresh = () => {
            if (!source.isConnected) {
                return;
            }
            const options = sourceOptions(source);
            const selected = selectedSourceOption(source, options);
            if (selected && !options.some(
                (option) => String(option.value) === String(source.value)
            )) {
                source.value = selected.value;
            }
            const selectedLabel = selected?.textContent || "Select";
            if (valueLabel.textContent !== selectedLabel) {
                valueLabel.textContent = selectedLabel;
            }
            if (trigger.disabled !== Boolean(source.disabled)) {
                trigger.disabled = Boolean(source.disabled);
            }
            if (trigger.getAttribute("aria-disabled") !== String(trigger.disabled)) {
                trigger.setAttribute("aria-disabled", String(trigger.disabled));
            }
            const invalid = source.getAttribute("aria-invalid");
            if (trigger.getAttribute("aria-invalid") !== invalid) {
                if (invalid === null) {
                    trigger.removeAttribute("aria-invalid");
                } else {
                    trigger.setAttribute("aria-invalid", invalid);
                }
            }
            // Input/change events and attribute observers often report the same
            // selection. Keep option nodes (and keyboard focus) unless the
            // source list itself changes, instead of rebuilding it each time.
            if (options.length !== renderedOptions.length || options.some(
                (option, index) => option !== renderedOptions[index]
            )) {
                renderedOptions = options;
                optionControls = options.map((option, index) => {
                    const control = document.createElement("button");
                    control.className = "rwkv-popup-listbox__option";
                    control.type = "button";
                    control.id = `${id}-option-${index}`;
                    control.setAttribute("role", "option");
                    control.addEventListener("click", () => {
                        if (control.disabled) {
                            return;
                        }
                        source.value = option.value;
                        refresh();
                        closePopupControl({ restoreFocus: true });
                        dispatchPopupSourceEvent(source, "input");
                        dispatchPopupSourceEvent(source, "change");
                    });
                    return control;
                });
                panel.replaceChildren();
                for (const control of optionControls) {
                    panel.appendChild(control);
                }
            }
            for (const [index, option] of options.entries()) {
                const control = optionControls[index];
                const isSelected = String(
                    selected === option || String(option.value) === String(source.value)
                );
                if (control.getAttribute("aria-selected") !== isSelected) {
                    control.setAttribute("aria-selected", isSelected);
                }
                if (control.disabled !== Boolean(option.disabled)) {
                    control.disabled = Boolean(option.disabled);
                }
                const text = option.textContent || String(option.value);
                if (control.textContent !== text) {
                    control.textContent = text;
                }
            }
        };
        record.refresh = refresh;
        popupControlRecords.set(source, record);
        refresh();
        source.addEventListener("input", refresh);
        source.addEventListener("change", refresh);
        source.addEventListener("focus", () => trigger.focus());
        trigger.addEventListener("click", () => {
            if (activePopupControl === record) {
                closePopupControl({ restoreFocus: true });
            } else {
                refresh();
                openPopupControl(record, { focusSelected: true });
            }
        });
        trigger.addEventListener("keydown", (event) => {
            if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
                event.preventDefault();
                refresh();
                openPopupControl(record, { focusSelected: true });
            }
        });
        panel.addEventListener("keydown", (event) => {
            if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
                return;
            }
            const options = Array.from(
                panel.querySelectorAll('[role="option"]:not([disabled])')
            );
            if (!options.length) {
                return;
            }
            event.preventDefault();
            const current = options.indexOf(document.activeElement);
            const next = event.key === "Home" ? 0
                : event.key === "End" ? options.length - 1
                : (Math.max(0, current) + (event.key === "ArrowDown" ? 1 : -1)
                    + options.length) % options.length;
            options[next].focus();
        });
    };

    const padTemporalPart = (value) => String(value).padStart(2, "0");

    const validCalendarDate = (year, month, day) => {
        if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1) {
            return false;
        }
        return day <= new Date(Date.UTC(year, month, 0)).getUTCDate();
    };

    const normalizeDateText = (raw) => {
        const match = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(String(raw).trim());
        if (!match) {
            return null;
        }
        const year = Number(match[1]);
        const month = Number(match[2]);
        const day = Number(match[3]);
        return validCalendarDate(year, month, day)
            ? `${String(year).padStart(4, "0")}-${padTemporalPart(month)}-${padTemporalPart(day)}`
            : null;
    };

    const normalizeTimeText = (raw) => {
        const match = /^(\d{1,2}):([0-5]\d)(?::([0-5]\d))?$/.exec(String(raw).trim());
        if (!match) {
            return null;
        }
        const hour = Number(match[1]);
        if (hour > 23) {
            return null;
        }
        return `${padTemporalPart(hour)}:${match[2]}${match[3] ? `:${match[3]}` : ""}`;
    };

    const normalizeTemporalText = (kind, raw) => {
        const text = String(raw || "").trim();
        if (!text) {
            return "";
        }
        if (kind === "date") {
            return normalizeDateText(text);
        }
        if (kind === "time") {
            return normalizeTimeText(text);
        }
        const match = /^(\d{4}-\d{1,2}-\d{1,2})[ T](\d{1,2}:[0-5]\d(?::[0-5]\d)?)$/.exec(text);
        if (!match) {
            return null;
        }
        const date = normalizeDateText(match[1]);
        const time = normalizeTimeText(match[2]);
        return date && time ? `${date}T${time}` : null;
    };

    const temporalDisplayValue = (kind, value) => kind === "datetime-local"
        ? String(value || "").replace("T", " ")
        : String(value || "");

    const temporalValidationMessage = (record, normalized) => {
        const raw = String(record.proxy.value || "").trim();
        if (!raw) {
            return record.required ? "Enter a value." : "";
        }
        if (normalized === null) {
            return record.kind === "date" ? "Enter a date as YYYY-MM-DD."
                : record.kind === "time" ? "Enter a time as HH:MM."
                : "Enter a date and time as YYYY-MM-DD HH:MM.";
        }
        if (record.minimum && normalized < record.minimum) {
            return `Use ${temporalDisplayValue(record.kind, record.minimum)} or later.`;
        }
        if (record.maximum && normalized > record.maximum) {
            return `Use ${temporalDisplayValue(record.kind, record.maximum)} or earlier.`;
        }
        return "";
    };

    const syncTemporalFromProxy = (record, eventType = "input") => {
        const normalized = normalizeTemporalText(record.kind, record.proxy.value);
        const error = temporalValidationMessage(record, normalized);
        record.proxy.setCustomValidity?.(error);
        record.proxy.setAttribute("aria-invalid", String(Boolean(error)));
        // Keep the literal draft and caret while typing (including partial or
        // invalid dates). Only normalize after the text edit is committed.
        if (eventType === "change" && !error) {
            record.proxy.value = temporalDisplayValue(record.kind, normalized);
        }
        record.writingSource = true;
        try {
            record.source.value = normalized === null ? "" : normalized;
            dispatchPopupSourceEvent(record.source, eventType);
        } finally {
            record.writingSource = false;
        }
        return !error;
    };

    const syncTemporalFromSource = (record, { syncValue = true } = {}) => {
        if (record.writingSource) {
            return;
        }
        if (syncValue) {
            record.proxy.value = temporalDisplayValue(record.kind, record.source.value);
        }
        record.proxy.disabled = Boolean(record.source.disabled);
        record.trigger.disabled = Boolean(record.source.disabled);
        const normalized = normalizeTemporalText(record.kind, record.proxy.value);
        const error = temporalValidationMessage(record, normalized);
        record.proxy.setCustomValidity?.(error);
        record.proxy.setAttribute("aria-invalid", String(
            Boolean(error) || record.source.getAttribute("aria-invalid") === "true",
        ));
    };

    const monthNames = (
        "January February March April May June July August September October November December"
    ).split(" ");
    const weekdayNames = "Sun Mon Tue Wed Thu Fri Sat".split(" ");

    const currentLocalDate = () => {
        const now = new Date();
        return `${String(now.getFullYear()).padStart(4, "0")}-${padTemporalPart(
            now.getMonth() + 1
        )}-${padTemporalPart(now.getDate())}`;
    };

    const commitTemporalPickerValue = (record, value) => {
        record.proxy.value = temporalDisplayValue(record.kind, value);
        syncTemporalFromProxy(record, "input");
        syncTemporalFromProxy(record, "change");
        closePopupControl({ restoreFocus: true });
    };

    const calendarButton = (label, className = "") => {
        const button = document.createElement("button");
        button.className = `rwkv-popup-calendar__button ${className}`.trim();
        button.type = "button";
        button.textContent = label;
        return button;
    };

    const renderTemporalPicker = (record) => {
        const panel = record.panel;
        panel.replaceChildren();
        const header = document.createElement("div");
        header.className = "rwkv-popup-calendar__header";
        const previous = calendarButton("‹", "rwkv-popup-calendar__nav");
        previous.setAttribute("aria-label", "Previous month");
        const title = document.createElement("strong");
        title.className = "rwkv-popup-calendar__title";
        title.textContent = `${monthNames[record.viewMonth - 1]} ${record.viewYear}`;
        const next = calendarButton("›", "rwkv-popup-calendar__nav");
        next.setAttribute("aria-label", "Next month");
        previous.addEventListener("click", () => {
            record.viewMonth -= 1;
            if (record.viewMonth < 1) {
                record.viewMonth = 12;
                record.viewYear -= 1;
            }
            renderTemporalPicker(record);
            positionPopupControl(record);
        });
        next.addEventListener("click", () => {
            record.viewMonth += 1;
            if (record.viewMonth > 12) {
                record.viewMonth = 1;
                record.viewYear += 1;
            }
            renderTemporalPicker(record);
            positionPopupControl(record);
        });
        header.appendChild(previous);
        header.appendChild(title);
        header.appendChild(next);
        panel.appendChild(header);

        const grid = document.createElement("div");
        grid.className = "rwkv-popup-calendar__grid";
        grid.setAttribute("role", "grid");
        for (const weekday of weekdayNames) {
            const heading = document.createElement("span");
            heading.className = "rwkv-popup-calendar__weekday";
            heading.textContent = weekday;
            grid.appendChild(heading);
        }
        const firstWeekday = new Date(
            Date.UTC(record.viewYear, record.viewMonth - 1, 1)
        ).getUTCDay();
        const daysInMonth = new Date(
            Date.UTC(record.viewYear, record.viewMonth, 0)
        ).getUTCDate();
        for (let blank = 0; blank < firstWeekday; blank += 1) {
            const spacer = document.createElement("span");
            spacer.className = "rwkv-popup-calendar__spacer";
            spacer.setAttribute("aria-hidden", "true");
            grid.appendChild(spacer);
        }
        for (let day = 1; day <= daysInMonth; day += 1) {
            const date = `${String(record.viewYear).padStart(4, "0")}-${padTemporalPart(
                record.viewMonth
            )}-${padTemporalPart(day)}`;
            const dayButton = calendarButton(String(day), "rwkv-popup-calendar__day");
            dayButton.setAttribute("role", "gridcell");
            dayButton.setAttribute("aria-label", date);
            dayButton.setAttribute("aria-selected", String(date === record.pendingDate));
            dayButton.classList.toggle("is-today", date === currentLocalDate());
            dayButton.addEventListener("click", () => {
                record.pendingDate = date;
                if (record.kind === "date") {
                    commitTemporalPickerValue(record, date);
                    return;
                }
                if (record.calendarTimeInput) {
                    record.pendingTime = record.calendarTimeInput.value;
                }
                renderTemporalPicker(record);
                positionPopupControl(record);
            });
            grid.appendChild(dayButton);
        }
        panel.appendChild(grid);

        const footer = document.createElement("div");
        footer.className = "rwkv-popup-calendar__footer";
        const today = calendarButton("Today", "rwkv-popup-calendar__today");
        today.addEventListener("click", () => {
            const value = currentLocalDate();
            record.pendingDate = value;
            const [year, month] = value.split("-").map(Number);
            record.viewYear = year;
            record.viewMonth = month;
            if (record.kind === "date") {
                commitTemporalPickerValue(record, value);
                return;
            }
            renderTemporalPicker(record);
            positionPopupControl(record);
        });
        footer.appendChild(today);
        if (record.kind === "datetime-local") {
            const timeLabel = document.createElement("label");
            timeLabel.className = "rwkv-popup-calendar__time-label";
            const timeText = document.createElement("span");
            timeText.textContent = "Time";
            const timeInput = document.createElement("input");
            timeInput.className = "rwkv-popup-calendar__time";
            timeInput.type = "text";
            timeInput.inputMode = "numeric";
            timeInput.placeholder = "HH:MM";
            timeInput.value = record.pendingTime || "00:00";
            timeLabel.appendChild(timeText);
            timeLabel.appendChild(timeInput);
            record.calendarTimeInput = timeInput;
            const apply = calendarButton("Apply", "rwkv-popup-calendar__apply");
            apply.addEventListener("click", () => {
                const time = normalizeTimeText(timeInput.value);
                if (!time) {
                    timeInput.setAttribute("aria-invalid", "true");
                    timeInput.focus();
                    return;
                }
                record.pendingTime = time;
                commitTemporalPickerValue(record, `${record.pendingDate}T${time}`);
            });
            timeInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    event.preventDefault();
                    apply.click();
                }
            });
            footer.appendChild(timeLabel);
            footer.appendChild(apply);
        }
        panel.appendChild(footer);
    };

    const prepareTemporalPicker = (record) => {
        const normalized = normalizeTemporalText(record.kind, record.proxy.value);
        const date = normalized && normalized.slice(0, 10) || currentLocalDate();
        const [year, month] = date.split("-").map(Number);
        record.pendingDate = date;
        record.pendingTime = record.kind === "datetime-local" && normalized
            ? normalized.slice(11)
            : "00:00";
        record.viewYear = year;
        record.viewMonth = month;
        record.calendarTimeInput = null;
        renderTemporalPicker(record);
    };

    const enhanceTemporalControl = (source) => {
        if (popupControlRecords.has(source) || source.hasAttribute("data-rwkv-native-popup")) {
            return;
        }
        const kind = String(source.type || source.getAttribute("type") || "").toLowerCase();
        if (!["date", "time", "datetime-local"].includes(kind)) {
            return;
        }
        const id = `rwkv-popup-${++popupControlSequence}`;
        const parent = source.parentElement;
        if (!parent) {
            return;
        }
        const required = Boolean(source.required || source.hasAttribute("required"));
        const minimum = source.getAttribute("min") || "";
        const maximum = source.getAttribute("max") || "";
        source.removeAttribute("required");
        source.removeAttribute("min");
        source.removeAttribute("max");
        // Validation belongs to the visible editor. Removing step restores the
        // browser's default minute step, which rejects otherwise valid typed
        // seconds in a hidden datetime-local source during form submission.
        source.setAttribute("step", "any");
        source.required = false;

        const wrapper = document.createElement("div");
        wrapper.className = `rwkv-popup-temporal rwkv-popup-temporal--${kind}`;
        wrapper.dataset.rwkvPopupOwner = id;
        parent.insertBefore(wrapper, source);
        wrapper.appendChild(source);
        source.classList.add("rwkv-popup-source");
        source.setAttribute("aria-hidden", "true");
        source.tabIndex = -1;

        const proxy = document.createElement("input");
        proxy.className = "rwkv-popup-temporal__input";
        for (const className of String(source.className || "").split(/\s+/).filter(Boolean)) {
            if (className !== "rwkv-popup-source") {
                proxy.classList.add(className);
            }
        }
        proxy.type = "text";
        proxy.id = `${id}-entry`;
        proxy.autocomplete = "off";
        proxy.spellcheck = false;
        proxy.value = temporalDisplayValue(kind, source.value);
        proxy.required = required;
        if (required) {
            proxy.setAttribute("required", "");
        }
        proxy.placeholder = source.getAttribute("placeholder")
            || (kind === "date" ? "YYYY-MM-DD"
                : kind === "time" ? "HH:MM"
                : "YYYY-MM-DD HH:MM");
        const describedBy = source.getAttribute("aria-describedby");
        if (describedBy) {
            proxy.setAttribute("aria-describedby", describedBy);
        }
        for (const attribute of ["data-rwkv-enter-action", "data-rwkv-enter-payload"]) {
            const value = source.getAttribute(attribute);
            if (value !== null) {
                proxy.setAttribute(attribute, value);
            }
        }
        proxy.setAttribute("aria-label", popupAccessibleLabel(source, "Choose a date and time"));
        wrapper.appendChild(proxy);
        transferInitialFocus(source, proxy);
        rewireExplicitLabels(source, proxy);

        const trigger = document.createElement("button");
        trigger.className = "rwkv-popup-temporal__trigger";
        trigger.type = "button";
        trigger.id = `${id}-trigger`;
        trigger.setAttribute("aria-expanded", "false");
        trigger.setAttribute("aria-haspopup", "dialog");
        trigger.setAttribute(
            "aria-label",
            kind === "date" ? "Open calendar" : "Open date and time picker",
        );
        trigger.textContent = "▦";
        if (kind === "time") {
            trigger.hidden = true;
            trigger.tabIndex = -1;
        }
        wrapper.appendChild(trigger);

        const record = {
            id,
            kind,
            source,
            wrapper,
            proxy,
            trigger,
            panel: null,
            required,
            minimum,
            maximum,
            writingSource: false,
            renderPanel: null,
        };
        const panel = createPopupPanel(
            record,
            "rwkv-popup-calendar",
            "dialog",
            kind === "date" ? "Choose a date" : "Choose a date and time",
        );
        record.panel = panel;
        record.renderPanel = () => prepareTemporalPicker(record);
        trigger.setAttribute("aria-controls", panel.id);
        popupControlRecords.set(source, record);
        syncTemporalFromSource(record);
        source.addEventListener("input", () => syncTemporalFromSource(record));
        source.addEventListener("change", () => syncTemporalFromSource(record));
        source.addEventListener("focus", () => proxy.focus());
        proxy.addEventListener("input", () => syncTemporalFromProxy(record, "input"));
        proxy.addEventListener("change", () => syncTemporalFromProxy(record, "change"));
        trigger.addEventListener("click", () => {
            if (activePopupControl === record) {
                closePopupControl({ restoreFocus: true });
            } else {
                openPopupControl(record);
            }
        });
    };

    const cleanupPopupControls = () => {
        for (const [source, record] of popupControlRecords.entries()) {
            if (source.isConnected) {
                continue;
            }
            if (activePopupControl === record) {
                closePopupControl();
            }
            record.panel.remove?.();
            popupControlRecords.delete(source);
        }
    };

    const initializePopupControls = (container = root) => {
        if (useInPagePopupControls) {
            for (const select of popupCandidates(container, "select")) {
                enhanceSelectControl(select);
            }
        }
        for (const input of popupCandidates(
            container,
            useInPagePopupControls
                ? "input[type=date], input[type=time], input[type=datetime-local]"
                : "input[type=date], input[type=datetime-local]",
        )) {
            enhanceTemporalControl(input);
        }
    };

    const syncPopupControl = (source) => {
        const record = popupControlRecords.get(source);
        if (!record) {
            return false;
        }
        if (record.kind === "select") {
            record.refresh();
        } else {
            syncTemporalFromSource(record);
        }
        return true;
    };

    const initializePopupControlWorkaround = () => {
        root.classList.toggle("rwkv-in-page-popup-controls", useInPagePopupControls);
        initializePopupControls(root);
        if (typeof window.MutationObserver === "function") {
            popupControlObserver = new window.MutationObserver((mutations) => {
                const changedControls = new Set();
                let removedNodes = false;
                for (const mutation of mutations) {
                    if (mutation.type === "childList") {
                        for (const node of mutation.addedNodes || []) {
                            initializePopupControls(node);
                        }
                        const select = mutation.target?.closest?.("select");
                        const record = popupControlRecords.get(select);
                        if (record) {
                            changedControls.add(record);
                        }
                        removedNodes ||= Boolean(mutation.removedNodes?.length);
                    } else if (mutation.type === "attributes") {
                        const record = popupControlRecords.get(mutation.target)
                            || popupControlRecords.get(mutation.target?.closest?.("select"));
                        if (record) {
                            changedControls.add(record);
                        }
                    }
                }
                // Updating many options in one turn should scan each select
                // once, not once per mutation (quadratic for large lists).
                for (const record of changedControls) {
                    if (record.kind === "select") {
                        record.refresh();
                    } else {
                        syncTemporalFromSource(record, { syncValue: false });
                    }
                }
                if (removedNodes) {
                    cleanupPopupControls();
                }
            });
            popupControlObserver.observe(root, {
                attributes: true,
                attributeFilter: ["aria-invalid", "disabled", "selected", "value"],
                childList: true,
                subtree: true,
            });
        }
        window.addEventListener("resize", () => positionPopupControl());
        window.addEventListener("scroll", () => positionPopupControl(), true);
    };

    const setDisclosureExpanded = (button, panel, expanded) => {
        button.setAttribute("aria-expanded", String(expanded));
        panel.hidden = !expanded;
        const label = button.querySelector("[data-rwkv-disclosure-label]");
        if (label) {
            const nextLabel = expanded
                ? label.dataset.rwkvExpandedLabel
                : label.dataset.rwkvCollapsedLabel;
            if (nextLabel !== undefined) {
                label.textContent = nextLabel;
            }
        }
        const expandedRootClass = String(button.dataset.rwkvExpandedRootClass || "").trim();
        if (expandedRootClass) {
            root.classList.toggle(expandedRootClass, expanded);
        }
    };

    const initializeDisclosures = () => {
        for (const button of root.querySelectorAll("[data-rwkv-disclosure]")) {
            const panel = document.getElementById(button.getAttribute("aria-controls") || "");
            if (!panel) {
                continue;
            }
            setDisclosureExpanded(button, panel, button.getAttribute("aria-expanded") === "true");
            button.addEventListener("click", () => {
                if (button.disabled || button.getAttribute("aria-disabled") === "true") {
                    return;
                }
                hideTooltip(button, { dismiss: true });
                const wasExpanded = button.getAttribute("aria-expanded") === "true";
                const expanded = !wasExpanded;
                setDisclosureExpanded(button, panel, expanded);
                const action = button.dataset.rwkvDisclosureAction;
                if (action) {
                    send(action, { expanded }).catch(() => {
                        setDisclosureExpanded(button, panel, wasExpanded);
                    });
                }
            });
        }
    };

    const configuredBackgroundSelectors = Array.isArray(bootstrap.backgroundSelectors)
        ? bootstrap.backgroundSelectors.filter((selector) => typeof selector === "string")
        : [];
    const backgroundRegions = configuredBackgroundSelectors.length
        ? Array.from(new Set(configuredBackgroundSelectors.flatMap(
            (selector) => Array.from(document.querySelectorAll(selector))
        )))
        : Array.from(root.children).filter((element) =>
            element.matches(".rwkv-modal-page, .rwkv-dialog-footer")
        );
    let backgroundRegionState = [];

    const setBackgroundInert = (inert) => {
        if (inert && !backgroundRegionState.length) {
            backgroundRegionState = backgroundRegions.map((region) => ({
                region,
                inert: Boolean(region.inert),
                ariaHidden: region.getAttribute("aria-hidden"),
            }));
        }
        if (!inert) {
            for (const entry of backgroundRegionState) {
                entry.region.inert = entry.inert;
                if (entry.ariaHidden === null) {
                    entry.region.removeAttribute("aria-hidden");
                } else {
                    entry.region.setAttribute("aria-hidden", entry.ariaHidden);
                }
            }
            backgroundRegionState = [];
            return;
        }
        for (const region of backgroundRegions) {
            region.inert = true;
            region.setAttribute("aria-hidden", "true");
        }
    };

    const showOverlay = (overlayOrId) => {
        const overlay = typeof overlayOrId === "string"
            ? document.getElementById(overlayOrId)
            : overlayOrId;
        if (!overlay) {
            return false;
        }
        closePopupControl();
        const originalFocus = focusBeforeOverlay || document.activeElement;
        if (activeOverlay && activeOverlay !== overlay) {
            hideOverlay(activeOverlay, { preserveStoredFocus: true, restoreFocus: false });
        }
        focusBeforeOverlay = originalFocus;
        overlay.hidden = false;
        overlay.setAttribute("aria-hidden", "false");
        activeOverlay = overlay;
        setBackgroundInert(true);
        const initial = overlay.querySelector("[data-rwkv-initial-focus]")
            || focusableElements(overlay)[0]
            || overlay.querySelector(".rwkv-overlay-panel");
        initial?.focus();
        return true;
    };

    const hideOverlay = (overlayOrId, options = {}) => {
        const overlay = typeof overlayOrId === "string"
            ? document.getElementById(overlayOrId)
            : overlayOrId;
        if (!overlay) {
            return false;
        }
        const wasActive = activeOverlay === overlay;
        overlay.hidden = true;
        overlay.setAttribute("aria-hidden", "true");
        if (wasActive) {
            activeOverlay = null;
            setBackgroundInert(false);
        }
        if (wasActive && options.restoreFocus !== false && focusBeforeOverlay?.isConnected) {
            focusBeforeOverlay.focus();
        }
        if (wasActive && options.preserveStoredFocus !== true) {
            focusBeforeOverlay = null;
        }
        return true;
    };

    const setProgress = ({
        overlayId = "rwkv-progress-overlay",
        token,
        title,
        label,
        current,
        total,
        eta,
        cancellable,
        cancelPending,
    }) => {
        const overlay = document.getElementById(overlayId);
        if (!overlay) {
            return false;
        }
        if (
            activeProgressToken !== null
            && token !== undefined
            && Number(token) !== activeProgressToken
        ) {
            return false;
        }
        const progress = overlay.querySelector('[role="progressbar"]');
        const bar = overlay.querySelector("[data-rwkv-progress-bar]");
        const value = overlay.querySelector("[data-rwkv-progress-value]");
        const etaElement = overlay.querySelector("[data-rwkv-progress-eta]");
        const titleElement = overlay.querySelector("[data-rwkv-progress-title]");
        const labelElement = overlay.querySelector("[data-rwkv-progress-label]");
        const determinate = Number.isFinite(Number(current)) && Number.isFinite(Number(total)) && Number(total) > 0;
        if (titleElement && title !== undefined) {
            titleElement.textContent = String(title);
        }
        if (labelElement && label !== undefined) {
            labelElement.textContent = String(label);
            progress?.setAttribute("aria-label", String(label));
        }
        if (determinate) {
            const bounded = Math.max(0, Math.min(Number(current), Number(total)));
            progress?.setAttribute("aria-valuemin", "0");
            progress?.setAttribute("aria-valuemax", String(total));
            progress?.setAttribute("aria-valuenow", String(bounded));
            if (bar) {
                bar.style.width = `${100 * bounded / Number(total)}%`;
            }
            if (value) {
                value.textContent = `${bounded} of ${total}`;
            }
        } else {
            progress?.removeAttribute("aria-valuemin");
            progress?.removeAttribute("aria-valuemax");
            progress?.removeAttribute("aria-valuenow");
            bar?.style.removeProperty("width");
            if (value) {
                value.textContent = "Working";
            }
        }
        const cancel = overlay.querySelector("[data-rwkv-overlay-cancel]");
        if (etaElement) {
            etaElement.textContent = cancelPending
                ? "Waiting for the current step to finish safely…"
                : String(eta || "ETA unknown");
        }
        if (cancel) {
            if (token !== undefined) {
                cancel.dataset.rwkvPayload = JSON.stringify({ token: Number(token) });
            }
            if (cancellable !== undefined) {
                cancel.hidden = !Boolean(cancellable);
            }
            if (cancelPending !== undefined) {
                cancel.disabled = Boolean(cancelPending);
                cancel.setAttribute("aria-disabled", String(Boolean(cancelPending)));
                cancel.textContent = cancelPending ? "Cancelling…" : "Cancel";
            }
        }
        return true;
    };

    const showProgress = (payload = {}) => {
        const overlayId = payload.overlayId || "rwkv-progress-overlay";
        activeProgressToken = Number(payload.token);
        setProgress({ ...payload, overlayId });
        const progressTarget = document.getElementById(overlayId);
        if (progressTarget?.hasAttribute("data-rwkv-inline-progress")) {
            progressTarget.hidden = false;
            progressTarget.setAttribute("aria-hidden", "false");
            progressTarget.setAttribute("aria-busy", "true");
            const initial = progressTarget.querySelector("[data-rwkv-initial-focus]");
            window.requestAnimationFrame(() => initial?.focus());
            return true;
        }
        return showOverlay(overlayId);
    };

    const hideProgress = (payload = {}) => {
        if (
            payload.token !== undefined
            && Number(payload.token) !== activeProgressToken
        ) {
            return false;
        }
        activeProgressToken = null;
        const overlayId = payload.overlayId || "rwkv-progress-overlay";
        const progressTarget = document.getElementById(overlayId);
        if (progressTarget?.hasAttribute("data-rwkv-inline-progress")) {
            progressTarget.hidden = true;
            progressTarget.setAttribute("aria-hidden", "true");
            progressTarget.setAttribute("aria-busy", "false");
            return true;
        }
        return hideOverlay(overlayId);
    };

    const showMessage = (payload = {}) => {
        const overlay = document.getElementById(payload.overlayId || "rwkv-message-overlay");
        if (!overlay || !Number.isFinite(Number(payload.token))) {
            return false;
        }
        const buttons = Array.isArray(payload.buttons) ? payload.buttons.slice(0, 3) : [];
        if (!buttons.length) {
            return false;
        }
        activeMessageToken = Number(payload.token);
        const tone = ["info", "success", "warning", "error"].includes(payload.tone)
            ? payload.tone
            : "info";
        overlay.dataset.rwkvOverlayKind = tone;
        const panel = overlay.querySelector("[data-rwkv-message-panel]");
        if (panel) {
            panel.classList.remove(
                "rwkv-overlay-panel--info",
                "rwkv-overlay-panel--success",
                "rwkv-overlay-panel--warning",
                "rwkv-overlay-panel--error",
            );
            panel.classList.add(`rwkv-overlay-panel--${tone}`);
        }
        const title = overlay.querySelector("[data-rwkv-message-title]");
        const message = overlay.querySelector("[data-rwkv-message-text]");
        const details = overlay.querySelector("[data-rwkv-message-details]");
        const checkboxContainer = overlay.querySelector(
            "[data-rwkv-message-checkbox-container]",
        );
        const checkbox = overlay.querySelector("[data-rwkv-message-checkbox]");
        const checkboxLabel = overlay.querySelector("[data-rwkv-message-checkbox-label]");
        if (title) {
            title.textContent = String(payload.title || "RWKV");
        }
        if (message) {
            if (typeof payload.messageHtml === "string") {
                // messageHtml is accepted only from Python's explicitly trusted,
                // add-on-owned renderer path. Ordinary/user text uses textContent.
                message.innerHTML = payload.messageHtml;
            } else {
                message.textContent = String(payload.message || "");
            }
        }
        if (details) {
            const hasDetails = typeof payload.details === "string" && payload.details.length > 0;
            details.hidden = !hasDetails;
            details.setAttribute("aria-hidden", String(!hasDetails));
            details.textContent = hasDetails ? payload.details : "";
        }
        const hasCheckbox = Boolean(payload.checkbox
            && typeof payload.checkbox.label === "string"
            && payload.checkbox.label.length > 0);
        if (checkboxContainer) {
            checkboxContainer.hidden = !hasCheckbox;
            checkboxContainer.setAttribute("aria-hidden", String(!hasCheckbox));
        }
        if (checkbox) {
            checkbox.checked = hasCheckbox && Boolean(payload.checkbox.checked);
            checkbox.disabled = !hasCheckbox;
        }
        if (checkboxLabel) {
            checkboxLabel.textContent = hasCheckbox ? payload.checkbox.label : "";
        }
        const controls = Array.from(overlay.querySelectorAll("[data-rwkv-message-button]"));
        controls.forEach((control, index) => {
            const button = buttons[index];
            control.removeAttribute("data-rwkv-initial-focus");
            control.removeAttribute("data-rwkv-overlay-cancel");
            if (!button) {
                control.hidden = true;
                control.setAttribute("aria-hidden", "true");
                control.removeAttribute("data-rwkv-payload");
                return;
            }
            const variant = ["primary", "secondary", "quiet", "destructive"].includes(button.variant)
                ? button.variant
                : "secondary";
            control.hidden = false;
            control.setAttribute("aria-hidden", "false");
            control.className = `rwkv-button rwkv-button--${variant}`;
            control.textContent = String(button.label || button.outcome || "Continue");
            control.dataset.rwkvPayload = JSON.stringify({
                token: activeMessageToken,
                outcome: String(button.outcome),
            });
            if (String(button.outcome) === String(payload.initialOutcome)) {
                control.setAttribute("data-rwkv-initial-focus", "");
            }
            if (String(button.outcome) === String(payload.escapeOutcome)) {
                control.setAttribute("data-rwkv-overlay-cancel", "");
            }
        });
        return showOverlay(overlay);
    };

    const hideMessage = (payload = {}) => {
        if (
            payload.token !== undefined
            && Number(payload.token) !== activeMessageToken
        ) {
            return false;
        }
        activeMessageToken = null;
        return hideOverlay(payload.overlayId || "rwkv-message-overlay");
    };

    const promptElements = (overlayId = "rwkv-prompt-overlay") => {
        const overlay = document.getElementById(overlayId);
        if (!overlay) {
            return null;
        }
        return {
            overlay,
            form: overlay.querySelector("[data-rwkv-prompt-form]"),
            title: overlay.querySelector("[data-rwkv-prompt-title]"),
            message: overlay.querySelector("[data-rwkv-prompt-message]"),
            label: overlay.querySelector("[data-rwkv-prompt-label]"),
            input: overlay.querySelector("[data-rwkv-prompt-input]"),
            textarea: overlay.querySelector("[data-rwkv-prompt-textarea]"),
            error: overlay.querySelector("[data-rwkv-prompt-error]"),
            cancel: overlay.querySelector("[data-rwkv-prompt-cancel]"),
            confirm: overlay.querySelector("[data-rwkv-prompt-confirm]"),
        };
    };

    const setPromptError = (elements, message = "") => {
        if (!elements?.error) {
            return;
        }
        const text = String(message || "");
        elements.error.textContent = text;
        elements.error.hidden = !text;
        elements.error.setAttribute("aria-hidden", String(!text));
        if (text) {
            announce(text);
        }
    };

    const setPromptBusy = (elements, busy) => {
        for (const control of [
            elements?.input,
            elements?.textarea,
            elements?.confirm,
        ]) {
            if (!control) continue;
            control.disabled = Boolean(busy);
            control.setAttribute("aria-disabled", String(Boolean(busy)));
        }
    };

    const finishPrompt = (accepted) => {
        const state = activePromptState;
        if (!state) {
            return false;
        }
        activePromptState = null;
        setPromptBusy(state.elements, false);
        setPromptError(state.elements);
        const value = accepted ? String(state.control.value || "") : "";
        hideOverlay(state.elements.overlay);
        state.resolve({ accepted: Boolean(accepted), value });
        return true;
    };

    const submitPrompt = async () => {
        const state = activePromptState;
        if (!state || state.busy) {
            return false;
        }
        const value = String(state.control.value || "");
        let errorMessage = "";
        if (state.options.required && !value.trim()) {
            errorMessage = String(state.options.requiredMessage || "Enter a value to continue.");
        } else if (
            Number.isFinite(Number(state.options.maxLength))
            && Number(state.options.maxLength) >= 0
            && value.length > Number(state.options.maxLength)
        ) {
            errorMessage = String(
                state.options.maxLengthMessage
                || `Use ${Number(state.options.maxLength)} characters or fewer.`
            );
        }
        if (errorMessage) {
            setPromptError(state.elements, errorMessage);
            state.control.focus();
            return false;
        }

        state.busy = true;
        setPromptBusy(state.elements, true);
        setPromptError(state.elements);
        try {
            if (typeof state.options.validate === "function") {
                const validation = await state.options.validate(value);
                if (typeof validation === "string" && validation) {
                    errorMessage = validation;
                } else if (validation === false) {
                    errorMessage = "The value is not valid.";
                }
            }
        } catch (error) {
            errorMessage = String(error?.message || error || "The value could not be validated.");
        }
        if (activePromptState !== state) {
            return false;
        }
        state.busy = false;
        setPromptBusy(state.elements, false);
        if (errorMessage) {
            setPromptError(state.elements, errorMessage);
            state.control.focus();
            return false;
        }
        return finishPrompt(true);
    };

    const showPrompt = (options = {}) => new Promise((resolve, reject) => {
        const elements = promptElements(options.overlayId || "rwkv-prompt-overlay");
        if (!elements || !elements.form || !elements.input || !elements.textarea) {
            reject(new Error("This page does not provide an editable prompt overlay."));
            return;
        }
        if (activePromptState || (activeOverlay && !activeOverlay.hidden)) {
            reject(new Error("Another dialog operation is already active."));
            return;
        }
        const multiline = Boolean(options.multiline);
        const control = multiline ? elements.textarea : elements.input;
        const otherControl = multiline ? elements.input : elements.textarea;
        control.hidden = false;
        control.setAttribute("aria-hidden", "false");
        control.setAttribute("data-rwkv-initial-focus", "");
        otherControl.hidden = true;
        otherControl.setAttribute("aria-hidden", "true");
        otherControl.removeAttribute("data-rwkv-initial-focus");
        control.value = String(options.value ?? "");
        if (options.placeholder) {
            control.setAttribute("placeholder", String(options.placeholder));
        } else {
            control.removeAttribute("placeholder");
        }
        if (Number.isFinite(Number(options.maxLength)) && Number(options.maxLength) >= 0) {
            control.setAttribute("maxlength", String(Math.trunc(Number(options.maxLength))));
        } else {
            control.removeAttribute("maxlength");
        }
        if (elements.title) {
            elements.title.textContent = String(options.title || "Enter a value");
        }
        if (elements.message) {
            const message = String(options.message || "");
            elements.message.textContent = message;
            elements.message.hidden = !message;
            elements.message.setAttribute("aria-hidden", String(!message));
        }
        if (elements.label) {
            elements.label.textContent = String(options.label || "Value");
        }
        if (elements.cancel) {
            elements.cancel.textContent = String(options.cancelLabel || "Cancel");
        }
        if (elements.confirm) {
            elements.confirm.textContent = String(options.confirmLabel || "Continue");
        }
        setPromptBusy(elements, false);
        setPromptError(elements);
        activePromptState = {
            busy: false,
            control,
            elements,
            options,
            resolve,
        };
        if (!showOverlay(elements.overlay)) {
            activePromptState = null;
            reject(new Error("The editable prompt could not be shown."));
            return;
        }
        control.focus();
        control.select?.();
    });

    for (const form of document.querySelectorAll("[data-rwkv-prompt-form]")) {
        form.addEventListener("submit", (event) => {
            event.preventDefault();
            submitPrompt();
        });
        form.closest("[data-rwkv-overlay]")
            ?.querySelector("[data-rwkv-prompt-cancel]")
            ?.addEventListener("click", () => finishPrompt(false));
    }

    const exclusiveBaseDisabled = new WeakMap();
    const exclusiveActive = (control) => control.type === "checkbox"
        ? Boolean(control.checked) : control.value !== "";
    const syncExclusiveField = (source, clearPeer = false) => {
        const peer = document.getElementById(source.dataset.rwkvExclusiveWith);
        if (!peer || peer.dataset.rwkvExclusiveWith !== source.id) return;
        if (clearPeer && !source.disabled && exclusiveActive(source)) {
            if (peer.type === "checkbox") peer.checked = false;
            else peer.value = "";
        }
        for (const [control, other] of [[source, peer], [peer, source]]) {
            control.disabled = exclusiveBaseDisabled.get(control) || exclusiveActive(other);
            control.setAttribute("aria-disabled", String(control.disabled));
        }
    };
    const initializeExclusiveFields = () => {
        const controls = document.querySelectorAll("[data-rwkv-exclusive-with]");
        for (const control of controls) {
            if (!exclusiveBaseDisabled.has(control)) {
                exclusiveBaseDisabled.set(control, Boolean(control.disabled));
            }
        }
        for (const control of controls) syncExclusiveField(control);
    };
    for (const kind of ["input", "change"]) {
        document.addEventListener(kind, (event) => {
            const source = event.target.closest("[data-rwkv-exclusive-with]");
            if (source && !source.disabled) syncExclusiveField(source, true);
        });
    }

    const serializeForm = (form) => {
        const payload = {};
        for (const [name, value] of new FormData(form).entries()) {
            if (Object.hasOwn(payload, name)) {
                payload[name] = Array.isArray(payload[name])
                    ? [...payload[name], value]
                    : [payload[name], value];
            } else {
                payload[name] = value;
            }
        }
        for (const checkbox of form.querySelectorAll('input[type="checkbox"][name]')) {
            // FormData omits disabled controls. Python still needs their visible
            // state when another action serializes the complete form.
            payload[checkbox.name] = Boolean(checkbox.checked);
        }
        for (const control of form.querySelectorAll(
            'input[name]:disabled:not([type="checkbox"]), select[name]:disabled, textarea[name]:disabled'
        )) {
            // Disabled dependency controls are omitted by FormData. Their
            // visible values are still part of the Python-owned draft and must
            // survive another control's change or a primary action.
            if (!Object.hasOwn(payload, control.name)) {
                payload[control.name] = control.value;
            }
        }
        // Multi-selects always carry an array, including zero/one selection and
        // disabled drafts. Inline listboxes retain native Ctrl/Shift behavior.
        for (const control of form.querySelectorAll('select[multiple][name]')) {
            payload[control.name] = sourceOptions(control)
                .filter((option) => option.selected && !option.disabled)
                .map((option) => option.value);
        }
        return payload;
    };

    const validateTemporalDrafts = (form, { report = false } = {}) => {
        for (const [source, record] of popupControlRecords) {
            if (!record.proxy || source.disabled || source.closest("form") !== form) continue;
            const normalized = normalizeTemporalText(record.kind, record.proxy.value);
            if (temporalValidationMessage(record, normalized)) {
                if (report) record.proxy.reportValidity?.();
                return false;
            }
        }
        return true;
    };

    const submitForm = (form, submitter = null, actionOverride = "") => {
        const action = actionOverride
            || submitter?.dataset?.rwkvAction
            || form.dataset.rwkvFormAction;
        submitter = submitter || findEnterSubmitter(form, action);
        if (submitter?.disabled || submitter?.getAttribute("aria-disabled") === "true") {
            return false;
        }
        if (!submitter?.formNoValidate && !form.reportValidity()) {
            return false;
        }
        const payload = {
            ...serializeForm(form),
            ...(submitter ? parsePayload(submitter) : {}),
        };
        send(action, payload).catch(() => {});
        return true;
    };

    const findEnterSubmitter = (form, action) => {
        const submitters = Array.from(form.elements || []).filter((control) => {
            const type = String(control.type || "").toLowerCase();
            return (
                (control.tagName === "BUTTON" && type === "submit")
                || (control.tagName === "INPUT" && ["submit", "image"].includes(type))
            );
        });
        const matching = submitters.filter((control) => control.dataset.rwkvAction === action);
        return matching.find((control) => !control.disabled
            && control.getAttribute("aria-disabled") !== "true") || matching[0] || null;
    };

    const activateTab = (tab) => {
        if (!tab || tab.disabled || tab.getAttribute("aria-disabled") === "true") {
            return false;
        }
        const tablist = tab.closest('[role="tablist"]');
        if (!tablist) {
            return false;
        }
        for (const candidate of tablist.querySelectorAll('[role="tab"]')) {
            const selected = candidate === tab;
            candidate.setAttribute("aria-selected", String(selected));
            candidate.tabIndex = selected ? 0 : -1;
            const panelId = candidate.getAttribute("aria-controls");
            const panel = panelId ? document.getElementById(panelId) : null;
            if (panel) {
                panel.hidden = !selected;
                panel.setAttribute("aria-labelledby", candidate.id);
            }
        }
        tab.dispatchEvent(new CustomEvent("rwkv:tab-activated", {
            bubbles: true,
            detail: { panelId: tab.getAttribute("aria-controls") },
        }));
        return true;
    };

    document.addEventListener("click", (event) => {
        if (
            activePopupControl
            && popupOwnerId(event.target) !== activePopupControl.id
        ) {
            closePopupControl();
        }
        if (!event.target.closest(".glossary-term")) {
            closeGlossaryTerms();
        }
        const tab = event.target.closest('[role="tab"]');
        if (tab) {
            event.preventDefault();
            activateTab(tab);
            return;
        }
        const button = event.target.closest("[data-rwkv-action]");
        if (!button || button.disabled || button.getAttribute("aria-disabled") === "true") {
            return;
        }
        if (button.type === "submit") {
            return;
        }
        event.preventDefault();
        const form = button.closest("form");
        if (button.hasAttribute("data-rwkv-serialize-form") && form
            && !button.formNoValidate && !validateTemporalDrafts(form, { report: true })) {
            return;
        }
        const payload = button.hasAttribute("data-rwkv-serialize-form") && form
            ? { ...serializeForm(form), ...parsePayload(button) }
            : parsePayload(button);
        send(button.dataset.rwkvAction, payload).catch(() => {});
    });

    document.addEventListener("submit", (event) => {
        const form = event.target.closest("form[data-rwkv-form-action]");
        if (!form) {
            return;
        }
        event.preventDefault();
        const submitter = event.submitter;
        submitForm(form, submitter);
    });

    function syncEnabledTargets(control, seen = new Set()) {
        if (seen.has(control)) return;
        seen.add(control);
        const dependencyTargets = String(control.dataset.rwkvEnableTargets || "")
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean);
        for (const targetId of dependencyTargets) {
            const target = document.getElementById(targetId);
            if (!target) {
                continue;
            }
            target.disabled = control.disabled || !Boolean(control.checked);
            if (target.disabled && target.type === "checkbox") target.checked = false;
            target.setAttribute("aria-disabled", String(target.disabled));
            syncEnabledTargets(target, seen);
        }
    }

    const checkAllBaseDisabled = new WeakMap();
    const checkAllTargets = (control) => String(control.dataset.rwkvCheckAllTargets || "")
        .split(",")
        .map(id => document.getElementById(id.trim()))
        .filter(target => target && target !== control && target.type === "checkbox"
            && !target.hasAttribute("data-rwkv-check-all-targets")
            && target.closest("form") === control.closest("form"));
    const syncCheckAllControls = () => {
        for (const control of document.querySelectorAll("[data-rwkv-check-all-targets]")) {
            if (!checkAllBaseDisabled.has(control)) {
                checkAllBaseDisabled.set(control, Boolean(control.disabled));
            }
            const targets = checkAllTargets(control).filter(target => !target.disabled);
            const checked = targets.filter(target => target.checked).length;
            control.checked = targets.length > 0 && checked === targets.length;
            control.indeterminate = checked > 0 && checked < targets.length;
            control.disabled = checkAllBaseDisabled.get(control) || !targets.length;
            control.setAttribute("aria-disabled", String(control.disabled));
            control.setAttribute("aria-checked", control.indeterminate ? "mixed" : String(control.checked));
        }
    };
    const changeCheckAll = (control) => {
        const pending = new Set(checkAllTargets(control));
        // Enabling a parent can unlock another target earlier in the list.
        // Revisit those targets without sending intermediate form/bridge actions.
        let changed = true;
        while (pending.size && changed) {
            changed = false;
            for (const target of pending) {
                if (target.disabled) continue;
                target.checked = Boolean(control.checked);
                syncEnabledTargets(target);
                pending.delete(target);
                changed = true;
            }
        }
    };

    document.addEventListener("change", (event) => {
        const all = event.target.closest("[data-rwkv-check-all-targets]");
        if (all && !all.disabled && all.getAttribute("aria-disabled") !== "true") {
            changeCheckAll(all);
            syncCheckAllControls();
            return;
        }
        const checkbox = event.target.closest('input[type="checkbox"]');
        if (checkbox && !checkbox.disabled) {
            syncEnabledTargets(checkbox);
            syncCheckAllControls();
        }
        const control = event.target.closest("[data-rwkv-change-action], [data-rwkv-enable-targets]");
        if (!control || control.disabled || control.getAttribute("aria-disabled") === "true") {
            return;
        }
        if (control !== checkbox) syncEnabledTargets(control);
        if (!control.dataset.rwkvChangeAction) return;
        const form = control.closest("form");
        let payload;
        if (control.dataset.rwkvChangeAction === "message-checkbox-change") {
            payload = {
                token: activeMessageToken,
                checked: Boolean(control.checked),
            };
        } else if (control.hasAttribute("data-rwkv-change-serialize-form") && form) {
            // Auto-updates can rebuild the whole page. Leave unfinished dates
            // editable, including when the user changes a sibling field.
            if (!validateTemporalDrafts(form)) return;
            payload = serializeForm(form);
        } else if (control.name) {
            payload = {
                [control.name]: control.type === "checkbox"
                    ? Boolean(control.checked)
                    : control.value,
            };
        } else {
            payload = {};
        }
        send(control.dataset.rwkvChangeAction, payload).catch(() => {});
    });

    document.addEventListener("keydown", (event) => {
        if (event.defaultPrevented) return;
        const rootKeyboardBlocked = Array.from(
            root.querySelectorAll("[data-rwkv-keyboard-scope]")
        ).some((scope) => !scope.hidden);
        if (event.key === "Escape") {
            if (activePopupControl) {
                event.preventDefault();
                closePopupControl({ restoreFocus: true });
                return;
            }
            if (activeTooltipTarget) {
                event.preventDefault();
                hideTooltip(activeTooltipTarget, { dismiss: true });
                return;
            }
            const glossaryTerm = event.target.closest(".glossary-term")
                || root.querySelector(".glossary-term.is-open");
            if (glossaryTerm) {
                event.preventDefault();
                closeGlossaryTerms();
                glossaryTerm.classList.remove("is-open");
                glossaryTerm.classList.add("tooltip-dismissed");
                glossaryTerm.setAttribute("aria-expanded", "false");
                return;
            }
            if (activeOverlay && !activeOverlay.hidden) {
                const cancel = activeOverlay.querySelector("[data-rwkv-overlay-cancel]");
                if (cancel && !cancel.disabled) {
                    event.preventDefault();
                    cancel.click();
                }
                return;
            }
            const action = rootKeyboardBlocked ? "" : root.dataset.rwkvEscapeAction;
            if (action) {
                event.preventDefault();
                let payload = {};
                try { payload = JSON.parse(root.dataset.rwkvEscapePayload || "{}"); } catch (_error) {}
                send(action, payload).catch(() => {});
            }
            return;
        }

        if (event.key === "Enter" && !event.isComposing && !activeOverlay) {
            const form = event.target.closest("form[data-rwkv-form-action]");
            if (form && event.target.matches("input:not([type=\"checkbox\"])") && !event.isComposing) {
                event.preventDefault();
                const action = event.target.dataset.rwkvEnterAction
                    || form.dataset.rwkvFormAction;
                const submitter = findEnterSubmitter(form, action);
                if (submitter?.disabled || submitter?.getAttribute("aria-disabled") === "true") {
                    return;
                }
                if (typeof form.requestSubmit === "function" && submitter) {
                    form.requestSubmit(submitter);
                } else if (
                    typeof form.requestSubmit === "function"
                    && action === form.dataset.rwkvFormAction
                ) {
                    form.requestSubmit();
                } else {
                    submitForm(form, submitter, action);
                }
                return;
            }
            const interactive = event.target.closest(
                'button, input, select, textarea, a[href], [contenteditable="true"]'
            );
            const inputCanSubmitRoot = root.dataset.rwkvEnterFromInputs === "true"
                && event.target.matches("input");
            const action = rootKeyboardBlocked ? "" : root.dataset.rwkvEnterAction;
            if ((!interactive || inputCanSubmitRoot) && action) {
                event.preventDefault();
                let payload = {};
                try { payload = JSON.parse(root.dataset.rwkvEnterPayload || "{}"); } catch (_error) {}
                send(action, payload).catch(() => {});
                return;
            }
        }

        if (event.key === "Tab" && activeOverlay && !activeOverlay.hidden) {
            const focusable = focusableElements(activeOverlay);
            if (!focusable.length) {
                event.preventDefault();
                activeOverlay.querySelector(".rwkv-overlay-panel")?.focus();
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

        const tab = event.target.closest('[role="tab"]');
        if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
            return;
        }
        const tabs = Array.from(
            tab.closest('[role="tablist"]')?.querySelectorAll('[role="tab"]:not([disabled])') || []
        );
        if (!tabs.length) {
            return;
        }
        event.preventDefault();
        const current = tabs.indexOf(tab);
        const next = event.key === "Home" ? 0
            : event.key === "End" ? tabs.length - 1
            : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        tabs[next].focus();
        tabs[next].click();
    });

    for (const selectedTab of document.querySelectorAll('[role="tablist"] [role="tab"][aria-selected="true"]')) {
        activateTab(selectedTab);
    }

    initializeGlossaryTerms();
    initializeTooltips();
    initializeDisclosures();
    initializeSortableTables();
    initializeExclusiveFields();
    syncCheckAllControls();
    // Local workflow scripts run immediately after this shared script and
    // may populate controls. Enhance dates everywhere, and Windows popup
    // controls when enabled, after the initial DOM has settled.
    window.requestAnimationFrame(initializePopupControlWorkaround);

    for (const overlay of document.querySelectorAll("[data-rwkv-overlay]:not([hidden])")) {
        showOverlay(overlay);
        break;
    }

    if (!activeOverlay) {
        window.requestAnimationFrame(() => {
            root.querySelector("[data-rwkv-initial-focus]:not([disabled])")?.focus();
        });
    }

    window.RWKVModal = Object.freeze({
        announce,
        activateTab,
        hideOverlay,
        initializeGlossaryTerms,
        initializeExclusiveFields,
        initializePopupControls,
        initializeSortableTables,
        initializeTooltips,
        send,
        serializeForm,
        syncCheckAllControls,
        setProgress,
        showOverlay,
        syncPopupControl,
    });
    window.RWKVProgress = Object.freeze({
        hide: hideProgress,
        show: showProgress,
        update: setProgress,
    });
    window.RWKVMessage = Object.freeze({
        hide: hideMessage,
        show: showMessage,
    });
    window.RWKVPrompt = Object.freeze({
        show: showPrompt,
    });
})();
