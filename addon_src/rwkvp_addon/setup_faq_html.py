from __future__ import annotations

import html

from .modal_html import ModalDisclosure, render_disclosure, render_notice

_FAQ_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "best-results",
        "How do I get the most out of RWKV4Anki?",
        f"""
<p>RWKV Immediate estimates how likely you are to remember each card. That
estimate changes as time passes and after every answer, so it works best when
cards are checked immediately before you review them.</p>
{render_notice(
    "Recommended: use RWKV Live Session for normal study.",
    tone="success",
)}
<p>On Anki’s <strong>Decks</strong> screen, click the <strong>gear icon</strong>
next to the deck you want to study, then choose
<strong>RWKV Live Session…</strong>. Live Session keeps the estimates current
and selects the next card for you.</p>
""".strip(),
    ),
    (
        "live-session-settings",
        "How should I configure a Live Session?",
        f"""
<p>Anki and FSRS normally assign cards future due dates. RWKV Immediate does
not. During a Live Session, it updates after each answer and checks which cards
need review <em>now</em>. It therefore does not load balance work across future
days.</p>
{render_notice(
    "Good starting point: if you normally do 100 reviews, set Minimum Reviews "
    "to 50 and Maximum Reviews to 150. These are guardrails, not daily targets.",
    tone="success",
)}
{render_notice(
    "Recommended sort orders: Ascending Retrievability or Relative Overdueness.",
    tone="info",
)}
<ul>
  <li><strong>Ascending Retrievability</strong> shows the card you are least
  likely to remember first.</li>
  <li><strong>Relative Overdueness</strong> prioritizes cards furthest below
  their deck’s desired-retention target.</li>
</ul>
""".strip(),
    ),
    (
        "minimum-reviews",
        "What is the purpose of Live Session’s Minimum Reviews?",
        f"""
<p>Minimum Reviews is a floor: when possible, Live Session keeps searching for
cards until you reach it.</p>
{render_notice(
    "Workload guardrail: it helps prevent a session from ending far below "
    "your usual workload, but it is not full load balancing.",
    tone="info",
)}
{render_notice(
    "Prediction safeguard: a lucky streak can make RWKV temporarily conclude "
    "that no cards are below your desired-retention target. Minimum Reviews "
    "keeps the session open long enough to gather more answers and correct "
    "that temporary overconfidence.",
    tone="warning",
)}
<p>RWKV updates its estimates after every answer. Until you reach the minimum,
Live Session temporarily widens card selection beyond the original retention
cutoff. The additional answers often recalibrate RWKV’s estimates and reveal
cards below your original target that an early exit would have missed.</p>
""".strip(),
    ),
    (
        "bad-day",
        "What if I’m having a really bad day?",
        f"""
<p>RWKV Immediate reacts to how you are performing right now. If you miss
cards you would normally remember, RWKV may predict that you are also more
likely to miss other cards. More of them can then fall below your desired
retention and enter the Live Session.</p>
{render_notice(
    "This is expected behavior: on a bad day, Live Session may grow because "
    "RWKV correctly detects that more cards currently need attention.",
    tone="warning",
)}
{render_notice(
    "Recommended: always set a Maximum Reviews limit and stop when you reach "
    "it. If you are clearly tired, ill, or distracted, it is also reasonable "
    "to stop early and continue another day.",
    tone="success",
)}
""".strip(),
    ),
    (
        "checkpoint",
        "What is a checkpoint?",
        f"""
{render_notice(
    "A checkpoint is RWKV’s saved memory of your collection. It is not an "
    "Anki backup and does not replace your collection.",
    tone="info",
)}
<p>RWKV reads your review history in order and builds information about your
cards, notes, decks, and presets. Saving that information as a checkpoint lets
RWKV load only what it needs instead of rebuilding everything whenever you use
an RWKV tool.</p>
""".strip(),
    ),
    (
        "checkpoint-consistency",
        "What is checkpoint consistency?",
        f"""
<p>A checkpoint is <strong>consistent</strong> when it still matches your review
history, including the order of reviews and the decks and presets involved.</p>
{render_notice(
    "A checkpoint can become inconsistent when older reviews sync after newer "
    "ones, a reviewed card is deleted or moved to another deck, or its deck "
    "changes options preset.",
    tone="warning",
)}
<p>These changes matter because RWKV learns from reviews in time order and also
uses deck and preset information. A mismatch can make its predictions less
predictable.</p>
{render_notice(
    "An inconsistent checkpoint will usually still work. Rebuilding gives "
    "RWKV the cleanest and most predictable state.",
    tone="success",
)}
""".strip(),
    ),
    (
        "other-devices",
        "How can I use RWKV on my other devices?",
        f"""
{render_notice(
    "RWKV4Anki does not run on AnkiMobile or AnkiDroid. Its checkpoint and "
    "predictions do not sync to those devices.",
    tone="warning",
)}
<p>As an optional workaround, create a filtered deck on your computer: on the
<strong>Decks</strong> screen, click the deck’s <strong>gear icon</strong>, open
<strong>RWKV (Immediate)</strong> or <strong>RWKV (Forgetting Curve)</strong>,
then choose <strong>Generate Filtered Deck…</strong>. After syncing, you can
study that filtered deck on another device.</p>
{render_notice(
    "Not recommended for normal use: filtered decks cannot update RWKV between "
    "reviews on another device and may behave unexpectedly. Although Forgetting "
    "Curve is designed to predict ahead, limited experiments found that "
    "Immediate often performed better when cards were selected in a batch this "
    "way.",
    tone="error",
)}
""".strip(),
    ),
    (
        "evaluation",
        "How can I see how well RWKV works for me?",
        f"""
<h3 class="setup-faq-answer-heading">Live Session results</h3>
<p>After each Live Session, its results table groups Same Day, Young, Mature,
and overall reviews. It compares what you actually remembered with the
predictions recorded immediately before each review. The difference columns
show how far RWKV and FSRS were from your actual recall, while
<strong>RWKV / FSRS Error</strong> compares the size of their errors.</p>
{render_notice(
    "Do not judge RWKV from one session. A short session is a small, noisy "
    "sample and can look surprisingly poor. Results collected over a week or "
    "longer are usually more stable and representative.",
    tone="warning",
)}
<p>Open <strong>RWKV → Live Review History…</strong> to combine saved sessions.
Its <strong>All Reviews</strong> table can show one week, one month, three
months, one year, or all saved reviews. You can also inspect the table for any
individual session.</p>
{render_notice(
    "Live Review History is useful for watching performance over time, but it "
    "is still a summary of Live Session results rather than the strongest test "
    "of RWKV’s predictive accuracy.",
    tone="info",
)}
<h3 class="setup-faq-answer-heading">Evaluation and calibration</h3>
<p>Open Anki’s <strong>RWKV</strong> menu, choose
<strong>Evaluate Immediate</strong> or <strong>Evaluate Forgetting Curve</strong>,
then choose <strong>Evaluate…</strong> and run the evaluation.</p>
<p>The results compare predicted recall with what you actually remembered:</p>
<ul>
  <li><strong>LogLoss</strong> measures the accuracy of individual predictions
  and penalizes confident mistakes.</li>
  <li><strong>RMSE(bins)</strong> checks whether groups of similar predictions
  match your actual recall rate.</li>
</ul>
<p>From the same menu, <strong>Calibration Graph…</strong> groups reviews by
their predicted recall and plots that prediction against what you actually
remembered. Results closer to the diagonal line are better calibrated.</p>
{render_notice(
    "Best indicator: use Evaluate and the Calibration Graph across many "
    "reviews. Lower Evaluate scores are better; if RWKV scores lower than "
    "FSRS-6 on the same reviews, RWKV matched that history better.",
    tone="success",
)}
""".strip(),
    ),
    (
        "forgetting-curve",
        "What about RWKV Forgetting Curve?",
        f"""
{render_notice(
    "Limited support: RWKV4Anki is developed primarily around RWKV Immediate. "
    "Forgetting Curve receives less integration work, testing, and support.",
    tone="warning",
)}
<p>You can still calculate, inspect, evaluate, and use forgetting curves, but
expect a less polished experience than RWKV Immediate.</p>
""".strip(),
    ),
    (
        "known-issues",
        "Known issues",
        f"""
<h3 class="setup-faq-answer-heading">Windows popup display bug</h3>
<p>On some Windows systems, opening a dropdown or date and time picker in an
RWKV4Anki window can make a blank white area repeatedly grow from the popup.
This issue has only been reported on Windows so far.</p>
{render_notice(
    "Workaround: open Settings → Advanced → Appearance and enable Windows "
    "Display Bug Patch. It uses in-window controls instead. Close and reopen "
    "any affected RWKV windows after changing it.",
    tone="success",
)}
<h3 class="setup-faq-answer-heading">A Live Session undo may need two attempts</h3>
<p>RWKV4Anki must reuse Anki’s filtered-deck machinery to run a Live Session
inside the normal reviewer and continually replace its next cards. The add-on
normally combines those filtered-deck updates with each answer’s undo entry,
but Undo can rarely encounter one of those updates before the answer.</p>
{render_notice(
    "If Undo appears to do nothing, leaves you on the current card, or shows a "
    "different card instead of the previous one, simply use Undo again. The "
    "second Undo should return to the review you intended to undo.",
    tone="warning",
)}
""".strip(),
    ),
)


def render_setup_faq_link() -> str:
    """Render the unobtrusive FAQ launcher shown on terminal setup pages."""

    return """
<div class="setup-faq-launch">
  <button type="button" class="rwkv-button setup-faq-launch-button"
          data-setup-faq-open aria-haspopup="dialog"
          aria-controls="rwkv-setup-faq-overlay">
    Learn more about using this add-on
  </button>
</div>
""".strip()


def render_setup_faq_help_button() -> str:
    """Render the compact launcher shown in the General Settings heading."""

    tooltip = "Learn more about using this add-on?"
    return f"""
<button type="button"
        class="rwkv-button rwkv-button--quiet setup-faq-help-button"
        id="rwkv-settings-faq-help"
        data-setup-faq-open data-rwkv-tooltip="{html.escape(tooltip, quote=True)}"
        aria-label="{html.escape(tooltip, quote=True)}"
        aria-haspopup="dialog" aria-controls="rwkv-setup-faq-overlay">
  <span aria-hidden="true">?</span>
</button>
""".strip()


def render_setup_faq_overlay() -> str:
    """Render the static, in-WebView FAQ with every answer collapsed."""

    sections = "".join(
        '<section class="setup-faq-item">'
        + render_disclosure(
            ModalDisclosure(
                button_id=f"rwkv-setup-faq-{slug}",
                panel_id=f"rwkv-setup-faq-{slug}-answer",
                collapsed_label=title,
                expanded_label=title,
                button_classes=("setup-faq-disclosure",),
                panel_classes=("setup-faq-answer",),
                panel_label=f"Answer: {title}",
            ),
            answer,
        )
        + "</section>"
        for slug, title, answer in _FAQ_SECTIONS
    )
    return f"""
<div class="rwkv-modal-overlay setup-faq-overlay"
     id="rwkv-setup-faq-overlay" hidden aria-hidden="true"
     role="dialog" aria-modal="true"
     aria-labelledby="rwkv-setup-faq-title"
     aria-describedby="rwkv-setup-faq-intro"
     data-rwkv-overlay data-rwkv-overlay-kind="info">
  <section class="rwkv-overlay-panel rwkv-overlay-panel--info setup-faq-panel"
           tabindex="-1">
    <header class="setup-faq-header">
      <div>
        <div class="setup-step-label">Getting started</div>
        <h2 class="rwkv-overlay-title" id="rwkv-setup-faq-title">
          Using RWKV4Anki
        </h2>
      </div>
      <button class="rwkv-icon-close" type="button"
              aria-label="Close using RWKV4Anki guide"
              data-setup-faq-close data-rwkv-overlay-cancel
              data-rwkv-initial-focus>&times;</button>
    </header>
    <div class="setup-faq-content">
      <p class="setup-copy" id="rwkv-setup-faq-intro">
        Short answers to common questions. Open any section to see its main
        recommendation and details.
      </p>
      <div class="setup-faq-list">{sections}</div>
    </div>
  </section>
</div>
""".strip()


__all__ = [
    "render_setup_faq_help_button",
    "render_setup_faq_link",
    "render_setup_faq_overlay",
]
