# Writing style for technical text

The rule for every word this repository ships: documentation, changelogs,
API docstrings, help text, error messages, and code comments. Prose should
read as if a careful engineer wrote it by hand, and a reader should
understand each sentence on the first pass.

## The target, in one paragraph

Direct, terse, exact. Simple words. Each abstraction paired with one
concrete detail (a type, a flag, a number). Honest about limits. Human
rhythm. No marketing, no padding, no clever asides, no AI slop.

## Three models, none copied strictly

The style leans on three sources. Take the discipline of each. Skip the
parts that would make text stiff.

1. Google developer documentation style, for how-to text (Quick start,
   Build, generate, serve). Second person ("you"), present tense,
   active voice, condition before instruction ("With `--kv q8`, the
   probe runs at load"), sentence-case headings, serial commas, American
   spelling, code font for identifiers and commands, unambiguous dates
   (2026-08-26). No "please". No "simply" or "just" as filler. No "we".
   Reference: https://developers.google.com/style/highlights

2. Simplified Technical English (ASD-STE100), as a lean, not the
   controlled dictionary. Short sentences (mostly under 25 words). One
   instruction per sentence. One topic per paragraph. One meaning per
   word and one word per meaning. No filler. Do not drop articles or verbs
   to save space. Noun clusters of at most three words.
   Reference: https://www.asd-ste100.org/ and
   https://en.wikipedia.org/wiki/Simplified_Technical_English

3. Mathematical exposition in the Halmos sense, for API, kernels, and
   receipts. Third person: "`Backend::allocate` rejects..." not "we
   reject..." and not "you get...". Say exactly what is true and nothing
   more. Define a term before you use it. Keep notation and terms
   consistent across the whole document set. State the limit of a claim as
   plainly as the claim. Prefer words to symbols in prose. No
   theorem-proof scaffolding.
   Reference: Halmos, "How to Write Mathematics" (1970).

## Sentences and words

- Short sentences, mostly under 25 words. Vary the rhythm a little; a page
  of identical sentences reads as machine output.
- Active voice, present tense. Second person only in how-to sections.
  API, kernel, and receipt text stay third person.
- Put the condition first, then the instruction.
- One term per concept, kept everywhere. Do not cycle synonyms.
  It is "decode" and "prefill" everywhere, never "generation phase" in one
  file and "token production" in another.
- Simple words: "use" not "utilize", "start" not "initiate", "make sure"
  not "ensure". Cut filler: "in order to", "it is important to note",
  "at this point in time".
- Do not drop articles or verbs for brevity. Keep noun clusters to three
  words at most.
- Serial commas. Sentence case in headings. American spelling. Code font
  for identifiers, flags, file names, and commands.
- No em dashes or en dashes. Use a period, a comma, a colon, or
  parentheses.
- No metaphors, allusions, jokes, or references to things outside the
  subject. No marketing words (robust, seamless, powerful, blazing,
  cutting-edge).
- Numbers carry their scope ("median 172.494 decode tok/s, five
  repetitions, depth 512"). Say median and maximum, not "within". Never
  type a measured number by hand. Copy it from the receipt under
  `receipts/`.

## Structure of an explanation

- Claim first, then mechanism, then limit.
  Example: "Decode is the gated workload. Prefill is tiled cuBLASLt with
  reusable FP16 weights. Prefill is reported in the receipt and is not
  gated against llama.cpp."
- Pair the abstraction with the concrete. Name the type, the function,
  the flag, the number.
- Define before use. Keep the same symbol or name for the same thing in
  every file (if `eta` is roofline efficiency, it is `eta` everywhere).
- Say what is not covered as plainly as what is.
- Prefer a table to a paragraph of numbers.

## Comments and docstrings

- A comment states a non-obvious constraint, invariant, or reason. It does
  not narrate what the code visibly does. If it does neither, delete it.
- No separator or banner comments.
- Docstrings: the first line is one sentence that says what the item is
  or does. Then constraints and defaults. Then links to related items.
- Error messages and help text follow the same rules as prose: short,
  exact, one meaning per word.

## Signs of machine-written text to remove

Em dashes; "not just X but Y"; groups of three for their own sake;
"serves as", "stands as", "is a testament to"; "delve", "leverage",
"robust", "seamless", "landscape", "underscore", "highlight" as a verb;
present-participle tails that add fake depth ("..., ensuring reliability");
vague authorities ("experts agree"); upbeat empty closings; sycophantic
openers; every sentence the same length; boldface on every other phrase;
bullet lists whose items all start with a bolded label and a colon.

## A quick checklist before you finish

1. Can each sentence be read once and understood? If not, split it.
2. Is every claim true as stated, with its scope attached?
3. Is every term used the same way as in the rest of the document set?
4. Does each abstraction come with one concrete example?
5. Any em dashes, marketing words, or filler left? Remove them.
6. Does it sound like a person wrote it? If it sounds like a template,
   vary the rhythm and cut the padding.
