# RWKV4Anki

RWKV4Anki brings recurrent spaced-repetition inference to Anki Desktop. It uses
[RWKV-SRS](https://github.com/zaguk/rwkv-srs) to maintain a compact recurrent
state for your review history, predict the probability that you will remember
cards, and update those predictions as you study.

The add-on bundles RWKV-SRS’s Rust/PyO3 backend and its thin Python facade. It
does not depend on Torch. Prediction and state building support optimized CPU
modes and optional GPU acceleration on compatible hardware.

## Attribution and AI authorship

RWKV4Anki and RWKV-SRS originated from the
[RWKV implementation](https://github.com/open-spaced-repetition/srs-benchmark/tree/main/rwkv)
in the Open Spaced Repetition
[`srs-benchmark`](https://github.com/open-spaced-repetition/srs-benchmark)
project, authored by [1DWalker](https://github.com/1DWalker). We gratefully
acknowledge that work as the foundation of this project.

**AI-authorship disclaimer:** This entire RWKV4Anki project—including its Anki
integration, user interface, checkpoint orchestration, tests, tooling,
documentation, and subsequent development—was coded by OpenAI’s Codex under
human direction. RWKV-SRS carries the same disclosure for its adaptation, Rust
implementation, Python bindings, tests, and tooling. This statement does not
supersede the attribution above for the original RWKV source on which these
projects were based.

Third-party attributions and exact locked dependency license texts for the
native runtime are included in packaged releases and recorded in RWKV-SRS’s
[`THIRD_PARTY_NOTICES.md`](https://github.com/zaguk/rwkv-srs/blob/main/THIRD_PARTY_NOTICES.md)
and
[`THIRD_PARTY_LICENSES.txt`](https://github.com/zaguk/rwkv-srs/blob/main/THIRD_PARTY_LICENSES.txt).

> **Pre-1.0 software:** RWKV-SRS’s API and checkpoint format may still change.
> RWKV4Anki pins an exact runtime version, but retaining normal backups while
> using pre-release software is recommended.

## Introduction to RWKV

[RWKV](https://github.com/BlinkDL/RWKV-LM) is a neural-network architecture that
combines useful properties of recurrent neural networks and Transformers. Like
an RNN, it can carry information forward in a compact recurrent state instead
of repeatedly rereading the complete history.

The Open Spaced Repetition benchmark adapted RWKV to review histories from the
Anki10k dataset. The model processes reviews in chronological order and can use
context shared across a collection, including cards, sibling notes, decks,
presets, answer time, and recent performance. Unlike algorithms that optimize a
separate parameter set for each user, the benchmark trained one shared model on
5,000 users and evaluated it on different users, repeating the split for full
coverage.

One trained recurrent model and state provide two useful kinds of prediction:

- **RWKV Immediate (RWKV-P)** estimates recall immediately before a review.
  Because it incorporates the latest answers from across the collection, it is
  most useful when predictions are refreshed while you study.
- **RWKV Forgetting Curve** estimates how recall changes with elapsed time. It
  supports more traditional predict-ahead reports and scheduling experiments.

In the benchmark’s published
[results without same-day reviews](https://github.com/open-spaced-repetition/srs-benchmark#without-same-day-reviews),
RWKV-P leads every practical spaced-repetition algorithm shown. It has the best
Log Loss, RMSE(bins), and AUC among schedulers; the table’s one lower
RMSE(bins) entry is explicitly a metric exploit with no practical scheduling
use. RWKV-P also leads the published table that includes same-day reviews.
Benchmark results describe aggregate predictive accuracy and do not guarantee
the same margin for every individual collection.

## Add-on features

- **RWKV Live Session:** refreshes predictions between reviews and selects
  cards whose estimated recall is below your desired retention.
- **Immediate and Forgetting Curve tools:** inspect Card Info, calculate average
  retrievability, and generate filtered decks from RWKV predictions.
- **Evaluation and calibration:** compare predictions with actual review
  outcomes using Log Loss, RMSE(bins), calibration graphs, and Live Session
  history.
- **CPU and GPU support:** choose optimized CPU modes or compatible GPU modes
  for prediction and large state builds. The setup wizard can benchmark the
  available choices on your collection.
- **Partial state loading:** study operations such as Live Session load only
  the relevant card, note, deck, and preset state from the complete checkpoint
  when possible, then release that in-memory state when the operation ends.
  This reduces RAM usage compared with keeping the entire collection state
  loaded while studying.
- **Local checkpoints:** preserve RWKV’s recurrent collection state so it does
  not need to be rebuilt whenever Anki starts.

RWKV4Anki requires Anki Desktop 26.05 or newer on Windows, macOS, or Linux. The
add-on manifest records 26.05 as the currently tested Anki release without
blocking later versions. RWKV4Anki does not run on AnkiMobile or AnkiDroid.
Filtered decks generated on a desktop can be synced to another device, but they
cannot update RWKV predictions between reviews there and are not recommended as
the normal way to use the add-on.

## Known issues

### Expanding popup on some Windows systems

On some Windows computers, opening a dropdown or date-and-time picker in an
RWKV4Anki window can make a blank white area repeatedly grow from the popup. If
this occurs, open **RWKV → Settings → Advanced → Appearance** and enable
**Windows Display Bug Patch**, then reopen the affected RWKV window.

### A Live Session undo may need two attempts

Live Session uses Anki’s filtered-deck machinery while continually updating its
next cards. Rarely, Undo encounters one of those deck updates before it reaches
the answer you intended to undo. If Undo appears to do nothing, leaves the
current card visible, or shows a different card, use Undo once more.

## How to install

1. Open this repository’s [Releases](../../releases) page.
2. Download the `.ankiaddon` file matching your operating system and CPU:

   | Computer | Release architecture |
   | --- | --- |
   | Most Windows and Linux PCs | `x86_64` |
   | Windows on ARM | `aarch64` |
   | Intel Mac | `x86_64` |
   | Apple Silicon Mac | `aarch64` |
   | ARM64 Linux | `aarch64` |

3. In Anki Desktop, open **Tools → Add-ons**, choose **Install from file…**, and
   select the downloaded file.
4. Restart Anki when prompted. RWKV4Anki’s setup wizard will help choose
   appropriate features and performance settings.

## How to package it yourself

The public source includes a standard-library Python packager. It requires
Python 3.11 or newer and produces a self-contained `.ankiaddon` in `dist/`.

From the repository root, package with the pinned official RWKV-SRS release for
the current computer:

```bash
python scripts/package_addon.py --version dev
```

The packager downloads only the exact RWKV-SRS wheel selected by
`rwkv-srs.lock.json` and verifies the release’s hashes and provenance. It also
downloads the two stable models declared by `rwkv-models.lock.json`, verifies
their SHA-256 values, and includes them in the resulting add-on. Anki does not
download the runtime or models after installation.

Official GitHub Releases build each architecture on a matching native runner.
They also include `SHA256SUMS` and `RELEASE_PROVENANCE.json`, which bind every
add-on bundle to this source export, the pinned RWKV-SRS release, and the fixed
model hashes.

To package with your own complete RWKV-SRS wheel:

```bash
python scripts/package_addon.py \
  --wheel /path/to/rwkv_srs.whl \
  --version dev
```

To combine your own native extension with a matching RWKV-SRS runtime wheel:

```bash
python scripts/package_addon.py \
  --native-extension /path/to/_native.abi3.so \
  --runtime-wheel /path/to/rwkv_srs.whl \
  --version dev
```

See the [RWKV-SRS repository](https://github.com/zaguk/rwkv-srs) for building a
portable wheel or a host-optimized native binary yourself. Run
`python scripts/package_addon.py --help` for target selection, offline caching,
local model overrides, and the complete packaging interface.

## Source availability

This repository is a generated distribution mirror containing the add-on
runtime and end-user packaging tools. Development tests, benchmarks, and
internal project records remain in the private canonical repository.

No license is granted for RWKV4Anki by this repository. Third-party notices
included with packaged releases apply only to their respective components.
