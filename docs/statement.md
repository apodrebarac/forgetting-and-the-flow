# The Forgetting and the Flow

*Degradation as Method in a Real-Time Image System*
Artist statement and technical appendix · TouchDesigner · 2026

## Artist Statement

This work is about what memory does to what it holds.

Its source is a private archive of images. But the images are not the work.
The work is what happens to them: two autonomous, real-time systems that
degrade, dissolve, and forget them continuously, in front of the viewer,
without ever repeating themselves.

The first system, *The Forgetting*, holds a memory that can only shrink. As
the images morph into one another, a mask of what is still remembered is
eaten away in small random bites, and by the slow growth of wounds already
open. Nothing the system does can restore what it loses. Damage done while
one image is visible scars every image that follows. Forgotten territory
becomes void. Forgiveness exists in this system as a single restoring
gesture, but it can only arrive from outside. Within its own dynamics,
forgetting is irreversible by construction.

The second system, *The Flow*, forgets differently. Each image is fed into an
endless liquid churn built entirely out of its own past states. Within
seconds the image as an image is gone; what persists is residue, its tones,
its weight, folded into a current that never settles and never returns to any
earlier state. The event disappears into its own retelling.

Neither system replays a recording. Both compute their decay live, seeded
fresh at every run, so that no exhibition, no viewing, no minute is
recoverable. The piece cannot remember what it looked like.

The forgetting has already begun. The system that generated the source images
no longer exists, it is destroyed, incompletely documented, unable to be
rebuilt. What survives is a finite archive of its outputs. Mortal images, in
the exact sense: no more can ever be made. The work's final form runs once.
On the day of its opening, the piece begins a single run of ten days and adds
one gesture to everything above: each image, every time it surfaces, is
written back to disk more degraded than it was loaded. A copy of a copy of a
copy. Generation by generation the archive consumes itself, until the files
hold noise, and then nothing. The forgiveness gesture is withheld for this
run. When it ends, there is no performance to replay and no archive to
reload. No copies are kept. The deletion of the last backup is performed by
the artist, by hand, before the run begins because that responsibility cannot
be automated. Most of the dying happens unwatched, at night, between
visitors; each audience inherits a smaller remainder and cannot feel what is
already missing. That, too, is how the memory of a person fades. Nobody
attends the whole forgetting.

The work is also a refusal. We live inside an economy of machine-made images
that only ever adds; infinite, weightless, costless. This machine is
generative in reverse. It can only subtract; it makes absence, at the rate of
one archive per existence. Against content that cannot die, it stages images
that can and do. Its entire output is one irreversible loss, made carefully.
Because building and running the final form are the same event, it cannot be
rehearsed, a rehearsal would already be the work. It exists now as this page,
and it will exist exactly once more.

## Technical Appendix

Both works are node-based real-time systems built in TouchDesigner, driven by
GPU texture operators (TOPs), with control logic in Python. Each is a
self-contained network fed by the full image archive: a shuffled deck with no
immediate repeats, advanced by a timer whose crossfades ping-pong between two
loader slots so the visible image never reloads. Everything below runs at 60
frames per second; nothing is pre-rendered.

### 1. The Flow: forgetting as lossy recursion

The Flow is a video feedback loop: each frame's output is written back as the
next frame's input. The loop body applies, in order: a sub-pixel zoom;
advection by a curl-noise vector field; self-refraction; a low-rate injection
of the source image; a slow hue rotation; and a clamp.

Every pass through the loop is lossy. The displacement operators resample the
texture with bilinear interpolation, discarding information below pixel
resolution; a mild diffusion averages neighbourhoods; values are clamped to
the displayable range. No buffer anywhere in the network stores the original
image — it exists only as a diminishing contribution inside the recursion.

That contribution can be quantified. With injection rate k, the source is
blended in each frame as `out = mix(loop, source, k)`, so after N frames the
original image's share of any pixel is approximately:

```
contribution(N) = 1 - (1 - k)^N        // k = 0.06
N = 11  ->  ~50% replaced      N = 75 (~1.25 s)  ->  ~99% replaced
```

An image is therefore mathematically half-forgotten in eleven frames and
effectively gone within a second and a quarter. What survives is not the
image but its statistics — dominant tones, energy distribution, palette —
which the loop keeps in circulation indefinitely.

**Motion without drift.** The velocity field is the curl of a scalar
simplex-noise field, computed in a GLSL shader by rotating the noise gradient
90 degrees. Because curl fields are divergence-free, the flow swirls without
any net directional drift and without sinks that would drain the canvas — the
churn is perpetual but goes nowhere.

**Self-refraction.** Each frame, the image is displaced by a blurred copy of
its own luminance — the picture refracts through itself. This continuously
re-forms crisp caustic folds out of material the loop would otherwise smear
flat, so the system generates structure and destroys it in the same gesture.

**Display grade, outside the loop.** A final shader applies a luminance-only
unsharp mask and a strong contrast expansion on the way to the screen only.
Placed inside the recursion these operations compound and destabilise
(posterisation, resonant hatching); outside it they are applied exactly once
per frame. The distinction matters conceptually as well: the deep blacks are
a way of seeing the decay, not part of the decay itself.

### 2. The Forgetting: forgetting as irreversible state

The second network implements forgetting as a data structure. A feedback loop
holds a single-channel texture — the memory mask — where white means
remembered. Its per-frame update rule is monotonic:

```
mask(t+1) = clamp( mask(t) - bites(t) - edge_creep(t), 0, 1 )
```

The mask is only ever subtracted from. In programming terms it is an
accumulator in a one-way state machine: no operation inside the system's
normal dynamics can increase it. The image stream is multiplied by the mask,
and lost regions render as void.

**Global, inherited damage.** The mask persists across image changes, so
erosion is global state rather than a per-image effect. Damage incurred while
one photograph is on screen is inherited by every photograph that follows —
the past of the system scars its future, which is the definition of path
dependence.

**Self-reinforcing loss.** Two processes subtract from the mask: sparse
random noise bites (reseeded for every image pair, so the decay pattern never
repeats), and edge creep, which erodes the mask preferentially at the borders
of existing voids. Loss compounds at its own margins — the computational
analogue of retrieval-induced forgetting, or of scar tissue.

**Surface degradation.** Above the erosion, a glitch layer keeps the image
itself unstable: horizontal slice bands tear sideways in discrete stutters, a
chunky dither converts tone to crawling stipple, and a black-and-white
contrast grade holds the material in a harsh forensic register. The image is
granular and never at rest even where it is still "remembered".

**Forgiveness as an external act.** The only operations that restore the mask
are a manual pulse labelled Reset Memory (forgive) and an optional rebirth
cycle for long installations. Both are interventions from outside the update
rule. Within the system's own dynamics, forgetting is irreversible by
construction.

### 3. Shared principles

**Statefulness and path dependence.** Both networks are stateful systems: the
present frame depends on the entire history of the run, not merely on current
inputs. The work at any moment is the sum of everything that has already
happened to it.

**Non-repeatability.** Deck order, noise seeds, and erosion patterns are
randomised per run and per transition. No two runs are identical and no state
is recoverable once passed — the system itself has no memory of its former
appearance. Each viewing is a unique, unrepeatable event.

**Entropy as content.** Both systems stage the second law in miniature:
highly structured information (the photograph) is converted into disorder
(marbled flow, void), and the conversion only runs one way. The images are not
displayed; they are processed — and the processing, not the picture, is the
work.

### 4. Glossary: system terms as memory terms

| System term | Memory analogue |
|---|---|
| Feedback loop | Recollection: each act of remembering takes the previous remembering, not the event, as its input. |
| Lossy resampling | The distortion introduced by every act of recall; detail below a certain grain is unrecoverable. |
| Injection rate | The waning voice of the original event inside its own retellings. |
| Exponential decay | How quickly the event itself, as opposed to its story, disappears. |
| Accumulator / monotonic state | Irreversibility: what is forgotten stays forgotten. |
| Edge creep | Loss breeding loss; forgetting spreading from the margins of what is already gone. |
| Random seeds, shuffled decks | Non-repeatability: no two rememberings, and no two viewings, are the same. |
| Manual reset ("forgive") | Recovery is possible only as an act from outside the system, never from within it. |

---

*This file carries the statement and the appendix of the work. The full
document, with its addendum on the resolution of the process, the general
artist statement and the oath, remains with the author.*
