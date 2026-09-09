# Intuition: AlphaGenome Atlas (Google DeepMind)

A simple way to understand the news.

## The camera flying into a cell

Imagine a DeepMind camera shrinking and flying into one human cell.

It passes through the cell, enters the **nucleus**, and finds **23 pairs of chromosomes**. Those are not loose noodles. They are the same DNA thread packed into tight, protective coils so it can fit.

The camera picks one chromosome and zooms until the coil unzips into a long chain. That chain is the DNA “sentence.”

A typical human genome has about **3 billion letters**. Each letter is one of four nucleotides: **A, T, C, or G**.

Most of those letters do not sit there making a protein by themselves. Only a small slice of the genome is the actual protein-recipe parts (genes). The rest is more like:

- switches
- volume knobs
- punctuation
- folding instructions

In other words: when to turn a gene on, how much RNA to make, how to cut the RNA, how tightly the DNA is packed, and which proteins grab onto that spot.

## What they did *not* do

They did **not** try every possible DNA sequence in the universe.

That is impossible. If you could freely rewrite all 3 billion letters, the number of possible genomes is (4^{3{,}000{,}000{,}000}). That number is so large it makes “all atoms in the universe” look tiny. Nobody computed that.

## What they actually did

They kept the existing human reference genome almost unchanged and asked only:

> At this one position, what if we change this one letter to one of the other three letters?

Do that at every position:

**3 billion positions × 3 other letters ≈ 9 billion single-letter changes.**

That set is complete *for that narrow job*: every possible one-letter substitution on the standard human sequence. It is still a tiny corner of “all DNA that could exist.”

### Book analogy

Think of the genome as one published book.

They did not write every possible book.

They generated every possible **single-letter typo** in *this* book, one typo at a time, then asked an AI:

> Does this typo break a sentence, change the chapter title, jam the printer, or do nothing?

## What the AI predicts

For each of those ~9 billion typos, **AlphaGenome** predicts molecular effects such as:

- Does nearby gene expression go up or down?
- Does RNA splicing change?
- Does DNA packing (chromatin) change?
- Do regulatory proteins still bind that spot?

They also added more than **100 million short insertions and deletions** that have actually been seen in real people, not just theoretical single-letter swaps.

Then they mashed many of those predictions into one number, the **AVI score** (AlphaGenome Variant Impact), so a researcher can quickly ask:

> Is this variant likely noisy, or worth a closer look?

## One-line summary

The camera’s target was not “invent new life codes.”

It was: take the human book we already have, try every single-letter misspelling, and predict which misspellings scramble the cell’s control panel.

## How this differs from AlphaFold

|                 |AlphaFold                                             |AlphaGenome Atlas                                                       |
|-----------------|------------------------------------------------------|------------------------------------------------------------------------|
|Object           |proteins                                              |DNA variants                                                            |
|Question         |What 3D shape does this amino-acid sequence fold into?|If I change one DNA letter, what happens to the molecular control panel?|
|Scale people cite|200+ million protein structures                       |9 billion single-letter DNA changes                                     |
|Ground truth     |often a crystal / experimental structure              |no single “correct shape”; many molecular readouts                      |

AlphaFold is closer to “what does the machine part look like?”  
AlphaGenome is closer to “what happens if we change one letter in the instruction manual?”

## Useful official links

- DeepMind blog: [AlphaGenome Atlas](https://deepmind.google/blog/alphagenome-atlas-a-predictive-map-of-every-possible-dna-letter-change-in-the-human-genome/)
- Nature news: [DeepMind’s new genome ‘atlas’](https://www.nature.com/articles/d41586-026-02835-4)
- Earlier model explainer: [AlphaGenome](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/)

## Relevant tweets

**Pushmeet Kohli (@pushmeet)**

- Atlas announcement (1/5): https://x.com/pushmeet/status/2097334856602517620
- AVI score (2/5): https://x.com/pushmeet/status/2097334859093930162
- Free access / AlphaFold Database parallel (3/5): https://x.com/pushmeet/status/2097334860603941084
- Rare disease / early use (4/5): https://x.com/pushmeet/status/2097334862873047459
- Reply to Eric Topol: https://x.com/pushmeet/status/2097418383126122735
- AlphaGenome in *Nature* (Jan 2026): https://x.com/pushmeet/status/2016550765842616755
- Original AlphaGenome launch (Jun 2025): https://x.com/pushmeet/status/1937873655888781647

**Eric Topol (@EricTopol)**

- The AlphaFold → AlphaGenome Atlas comparison post: https://x.com/erictopol/status/2097349624533299250



```
