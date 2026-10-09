# %% [markdown]
# # MiniGPT, rewritten to be read
#
# This program runs my already-trained MiniGPT model, the "exhibit" from
# [MiniGPT (Part 1)](https://haddley.github.io/posts/minigpt/), and nothing else: no training.
# It gives exactly the same answers as Jibin Joseph's original notebook code (section 9 checks this),
# and it is written to be read by a person.
#
# The rules I followed:
#
# - **Every value gets its own name.** There is no `x = x + something`. When a step makes something
#   new, the new thing gets a new name that says what it is.
# - **Every shape is written down.** A comment such as `# [positions, 128]` means "one row per letter
#   of the text, 128 numbers in each row".
# - **Every library call is explained** the first time it appears: what it does to the numbers.
# - **One text at a time.** The original code can work on a batch of texts at once, which adds an
#   extra dimension to every table. This version handles one text, so the tables are simpler.
# - **No hype.** Where nobody really knows *why* a piece works, I say so.

# %%
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch  # PyTorch: a library for tables of numbers ("tensors") and fast arithmetic on them

# %% [markdown]
# ## 1. Settings chosen before training
#
# These numbers were chosen by a person before the model was trained. Training never changes them,
# and the trained numbers only fit a model of exactly this shape.

# %%
NUMBERS_PER_VECTOR = 128            # every letter is represented by a list (a "vector") of 128 numbers
NUMBER_OF_BLOCKS = 4                # the model runs its hidden states through 4 blocks, one after another
HEADS_PER_BLOCK = 4                 # each block's attention runs 4 times side by side: 4 "head slices"
NUMBERS_PER_HEAD = NUMBERS_PER_VECTOR // HEADS_PER_BLOCK   # so each head works with 128 / 4 = 32 numbers
MOST_LETTERS_THE_MODEL_CAN_SEE = 128   # the "context length": positions beyond this have no embedding
MLP_WIDTH = 4 * NUMBERS_PER_VECTOR  # inside each MLP, 128 numbers are widened to 512, then narrowed back
TINY_NUMBER_TO_AVOID_DIVIDING_BY_ZERO = 0.00001   # used when normalising (PyTorch's default)

# The vocabulary: every different character in Tiny Shakespeare, in the computer's standard order.
# A letter's ID is simply its place in this string, counting from 0.
VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTER_TO_ID = {letter: place for place, letter in enumerate(VOCABULARY)}

print("letters in the vocabulary:", len(VOCABULARY))
print("ID of 'g':", LETTER_TO_ID["g"], "  ID of 'o':", LETTER_TO_ID["o"], "  ID of 'd':", LETTER_TO_ID["d"])

# %% [markdown]
# ## 2. The trained numbers: everything the model knows
#
# Training produced 826,433 numbers, and they are all the model knows. They are stored in a file,
# `exhibit.pt`, as named tables. `torch.load` reads the file back into a Python dictionary whose
# keys are the original code's names (such as `blocks.0.attn.query.weight`) and whose values are
# tensors: tables of numbers with a fixed shape.
#
# The original names are short and hard to read, so the next cell gives each table a descriptive
# name, using two small Python classes (`dataclass` just means "a record with named fields").
#
# None of these numbers changes while the model writes. They are the model's **static data**.

# %%
EXHIBIT_URL = "https://haddley.github.io/minigpt-demo/exhibit.pt"
EXHIBIT_FILE = Path("exhibit.pt")
if not EXHIBIT_FILE.exists():
    urllib.request.urlretrieve(EXHIBIT_URL, EXHIBIT_FILE)   # download the 3.4 MB file once

stored_tables = torch.load(EXHIBIT_FILE, map_location="cpu")   # a dictionary: name -> tensor


@dataclass
class TrainedBlock:
    """The trained numbers of one block. Every block has the same shapes, but its own values."""

    # Normalising before attention: one "stretch" and one "shift" number for each of the 128 positions
    # in a vector. (The original code calls these ln1.weight and ln1.bias.)
    stretch_before_attention: torch.Tensor      # [128]
    shift_before_attention: torch.Tensor        # [128]

    # Attention makes three new vectors from each hidden state: a looking query, a looked-at key, and a
    # passed-on value (usually just called the query, the key, and the value; section 6 explains the
    # extra words). Each comes from its own table of weights: 128 rows of 128 numbers, plus 128 biases.
    # (attn.query, attn.key, attn.value)
    looking_query_weights: torch.Tensor                 # [128, 128]
    looking_query_biases: torch.Tensor                  # [128]
    looked_at_key_weights: torch.Tensor                   # [128, 128]
    looked_at_key_biases: torch.Tensor                    # [128]
    passed_on_value_weights: torch.Tensor                 # [128, 128]
    passed_on_value_biases: torch.Tensor                  # [128]

    # After the four head slices have each collected 32 numbers, this table mixes the 128 numbers they
    # found into the 128 numbers that are added to the hidden state. (attn.proj)
    head_mixing_weights: torch.Tensor           # [128, 128]
    head_mixing_biases: torch.Tensor            # [128]

    # Normalising before the MLP. (ln2.weight, ln2.bias)
    stretch_before_mlp: torch.Tensor            # [128]
    shift_before_mlp: torch.Tensor              # [128]

    # The MLP: widen 128 numbers to 512, bend them, narrow them back to 128. (mlp.fc1, mlp.fc2)
    widen_weights: torch.Tensor                 # [512, 128]
    widen_biases: torch.Tensor                  # [512]
    narrow_weights: torch.Tensor                # [128, 512]
    narrow_biases: torch.Tensor                 # [128]


@dataclass
class TrainedModel:
    """All 826,433 trained numbers, under descriptive names."""

    # One row of 128 numbers for each of the 65 letters: the letter's "token embedding".
    token_embedding_table: torch.Tensor         # [65, 128]
    # One row of 128 numbers for each of the 128 positions: the position's "position embedding".
    position_embedding_table: torch.Tensor      # [128, 128]
    # The four blocks, in the order they run.
    blocks: list
    # The final normalisation, applied to the last hidden state. (final_ln)
    final_stretch: torch.Tensor                 # [128]
    final_shift: torch.Tensor                   # [128]
    # One row of 128 numbers, plus one bias, for each letter that could come next. (lm_head)
    next_letter_rows: torch.Tensor              # [65, 128]
    next_letter_biases: torch.Tensor            # [65]


def name_the_stored_tables(tables: dict) -> TrainedModel:
    blocks = []
    for block_number in range(NUMBER_OF_BLOCKS):
        prefix = f"blocks.{block_number}."
        blocks.append(TrainedBlock(
            stretch_before_attention=tables[prefix + "ln1.weight"],
            shift_before_attention=tables[prefix + "ln1.bias"],
            looking_query_weights=tables[prefix + "attn.query.weight"],
            looking_query_biases=tables[prefix + "attn.query.bias"],
            looked_at_key_weights=tables[prefix + "attn.key.weight"],
            looked_at_key_biases=tables[prefix + "attn.key.bias"],
            passed_on_value_weights=tables[prefix + "attn.value.weight"],
            passed_on_value_biases=tables[prefix + "attn.value.bias"],
            head_mixing_weights=tables[prefix + "attn.proj.weight"],
            head_mixing_biases=tables[prefix + "attn.proj.bias"],
            stretch_before_mlp=tables[prefix + "ln2.weight"],
            shift_before_mlp=tables[prefix + "ln2.bias"],
            widen_weights=tables[prefix + "mlp.fc1.weight"],
            widen_biases=tables[prefix + "mlp.fc1.bias"],
            narrow_weights=tables[prefix + "mlp.fc2.weight"],
            narrow_biases=tables[prefix + "mlp.fc2.bias"],
        ))
    return TrainedModel(
        token_embedding_table=tables["token_embedding.weight"],
        position_embedding_table=tables["position_embedding.weight"],
        blocks=blocks,
        final_stretch=tables["final_ln.weight"],
        final_shift=tables["final_ln.bias"],
        next_letter_rows=tables["lm_head.weight"],
        next_letter_biases=tables["lm_head.bias"],
    )


model = name_the_stored_tables(stored_tables)

# %% [markdown]
# Here is every table in the file, with its shape and how many numbers it holds. `tensor.shape` is a
# table's size in each direction, and `tensor.numel()` ("number of elements") multiplies them out.
#
# The file also holds four tables called `causal_mask`, one per block: 128 × 128 True or False
# values meaning "position *i* may look at position *j*". They are not learned numbers, just a rule
# written out as a table, and this program builds the same rule itself when it needs it (section 6).

# %%
learned_total = 0
for original_name, table in stored_tables.items():
    if original_name.endswith("causal_mask"):
        print(f"{original_name:30} {str(list(table.shape)):18} a fixed rule, not learned")
        continue
    learned_total += table.numel()
    print(f"{original_name:30} {str(list(table.shape)):18} {table.numel():>7,} numbers")
print(f"\nlearned numbers in all: {learned_total:,}")

# %% [markdown]
# A few of them, to show that they really are just numbers. Nobody wrote these by hand: training
# started them as small random numbers and nudged them, millions of times, towards whatever made the
# model better at guessing the next letter. None of the 128 numbers in a row has a meaning anyone
# assigned to it.

# %%
print("first 4 numbers of the 'g' token embedding:      ",
      model.token_embedding_table[LETTER_TO_ID["g"]][:4])
print("first 4 numbers of the position 1 embedding:     ",
      model.position_embedding_table[0][:4])
print("first 4 numbers of the 'd' next-letter row:      ",
      model.next_letter_rows[LETTER_TO_ID["d"]][:4])
print("the 'd' next-letter bias:                        ",
      model.next_letter_biases[LETTER_TO_ID["d"]])

# %% [markdown]
# `table[45]` picks row 45 of a table, and `[:4]` keeps only its first 4 numbers. Python counts from 0,
# so position 1 of the text is row 0 of the position table.
#
# ## 3. Four small tools
#
# The whole model is built from four simple operations, used again and again.
#
# ### Tool 1: applying a table of weights
#
# PyTorch calls this a "linear layer". For each input vector, every output number is a **dot product**:
# multiply the input's numbers by one row of weights, number by number, add up the results, and add
# that row's bias. A table with 512 rows turns each 128-number vector into 512 numbers.
#
# `weights.T` is the table turned on its side ("transposed"): rows become columns. The `@` sign is
# matrix multiplication, which here does every one of those dot products at once: every input row
# against every weight row.

# %%
def apply_weights(input_vectors: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """input_vectors: [rows, inputs]; weights: [outputs, inputs]; biases: [outputs] -> [rows, outputs]"""
    dot_products = input_vectors @ weights.T     # [rows, outputs]: each input row · each weight row
    return dot_products + biases                 # the same biases are added to every row


# %% [markdown]
# ### Tool 2: normalising
#
# "Layer normalisation" puts each vector's 128 numbers on a standard scale: an average of 0 and a
# spread of 1. Then each of the 128 numbers is stretched and shifted by its own trained amount.
#
# Why does this help? It keeps the numbers from growing or shrinking out of control as they pass
# through many steps, which makes training more stable. That much is well established; exactly why it
# helps as much as it does is still argued about. A 2019 study says "it is still unclear where the
# effectiveness stems from" (Xu and others, https://arxiv.org/abs/1911.07013).
#
# `.mean(dim=-1, keepdim=True)` takes the average along the last direction of the table, across the
# 128 numbers of each row, and keeps the answer as a one-number column so that it can be subtracted
# from every number in its row. `torch.sqrt` is the square root.

# %%
def normalise(vectors: torch.Tensor, stretch: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """vectors: [rows, 128] -> [rows, 128], each row rescaled to average 0 and spread 1, then stretched and shifted."""
    average_of_each_row = vectors.mean(dim=-1, keepdim=True)                       # [rows, 1]
    distance_from_average = vectors - average_of_each_row                          # [rows, 128]
    average_squared_distance = (distance_from_average ** 2).mean(dim=-1, keepdim=True)   # [rows, 1]
    spread_of_each_row = torch.sqrt(average_squared_distance + TINY_NUMBER_TO_AVOID_DIVIDING_BY_ZERO)
    standardised = distance_from_average / spread_of_each_row                      # [rows, 128]
    return standardised * stretch + shift                                          # [rows, 128]


# %% [markdown]
# ### Tool 3: the bend (GELU)
#
# Two tables of weights in a row, with nothing in between, would be no more powerful than one table,
# because a weighted mix of weighted mixes is still just a weighted mix. A bend between them fixes
# that. GELU ("Gaussian Error Linear Unit") lets big positive numbers through almost unchanged, turns
# big negative numbers into almost 0, and curves smoothly in between.
#
# `torch.erf` is the "error function", a standard S-shaped curve from statistics. This is the exact
# formula PyTorch uses for GELU. Why this curve rather than a simpler one? Mostly because it worked
# well in experiments (https://arxiv.org/abs/1606.08415). One paper on alternatives ends: "We offer no
# explanation as to why these architectures seem to work; we attribute their success, as all else, to
# divine benevolence" (Shazeer, https://arxiv.org/abs/2002.05202).

# %%
def bend(numbers: torch.Tensor) -> torch.Tensor:
    """GELU, applied to every number separately. Same shape in, same shape out."""
    return 0.5 * numbers * (1.0 + torch.erf(numbers / math.sqrt(2.0)))


print("bend(-3) =", round(bend(torch.tensor(-3.0)).item(), 4),
      "  bend(0) =", bend(torch.tensor(0.0)).item(),
      "  bend(3) =", round(bend(torch.tensor(3.0)).item(), 4))

# %% [markdown]
# ### Tool 4: turning scores into shares (softmax)
#
# Softmax turns a row of scores, which can be any size, even negative, into shares that are all
# positive and add up to 1. It is used twice in the model: to share out attention, and at the very end
# to turn the 65 scores into the 65 chances of the wheel.
#
# 1. Take the biggest score away from every score. This keeps the numbers small and does not change the
#    answer.
# 2. Raise *e* (about 2.718) to the power of each one, with `torch.exp`. The results are all positive,
#    and a bigger score always gives a bigger result. A score of minus infinity gives exactly 0.
# 3. Divide each result by the row's total, so that the row adds up to 1.
#
# `.amax(dim=-1, keepdim=True)` finds the biggest number in each row; `.sum(...)` adds each row up.

# %%
def scores_to_shares(scores: torch.Tensor) -> torch.Tensor:
    """Softmax along the last direction: each row of scores becomes shares that add up to 1."""
    biggest_score_in_each_row = scores.amax(dim=-1, keepdim=True)
    scores_minus_biggest = scores - biggest_score_in_each_row      # the biggest becomes 0
    positive_numbers = torch.exp(scores_minus_biggest)              # all between 0 and 1
    total_of_each_row = positive_numbers.sum(dim=-1, keepdim=True)
    return positive_numbers / total_of_each_row


print("shares for scores [2, 1, 0]:", scores_to_shares(torch.tensor([2.0, 1.0, 0.0])))

# %% [markdown]
# ## 4. The model's changing state
#
# Everything in section 2 is fixed. What changes is worked out fresh for each text, and thrown away
# afterwards. For a text of *n* letters, the model's state is:
#
# | Name in this program | Shape | What it is |
# |---|---|---|
# | `letter_ids` | [*n*] | each letter's ID |
# | `hidden_states` | [*n*, 128] | one vector per letter: it starts as token embedding + position embedding, and every block adds to it |
# | `looking_queries`, `looked_at_keys`, `passed_on_values` | [*n*, 128] each | made inside attention, from the hidden states; each head slice uses 32 of the 128 numbers |
# | `attention_shares` | [*n*, *n*] for each head slice | row *i*: how much of its attention position *i* gives to each position up to itself |
# | `widened` | [*n*, 512] | inside the MLP, between widening and narrowing |
# | `final_hidden_state` | [128] | the last letter's hidden state, after all four blocks and the final normalisation |
# | `scores` | [65] | one score (a "logit") for each letter that could come next |
# | `chances` | [65] | the scores after temperature, top-k and softmax: the wheel |
#
# Nothing else is remembered. The model has no memory between one letter and the next except the
# text itself: to write the next letter, it starts again from the letters.
#
# ## 5. From letters to the first hidden states
#
# Each letter's ID picks its row of the token-embedding table, and each position picks its row of the
# position-embedding table. Adding the two, number by number, gives each letter its first hidden
# state: what the letter is, and where it sits.
#
# `table[list_of_ids]` picks several rows at once, one for each ID in the list, in order.
# `table[:n]` keeps the first *n* rows.

# %%
def letters_to_ids(text: str) -> list:
    """Each letter's place in the vocabulary. Letters that are not in it are skipped."""
    return [LETTER_TO_ID[letter] for letter in text if letter in LETTER_TO_ID]


def first_hidden_states(letter_ids: list, model: TrainedModel) -> torch.Tensor:
    """[n] IDs -> [n, 128]: token embedding + position embedding, one row per letter."""
    number_of_letters = len(letter_ids)
    token_embeddings = model.token_embedding_table[letter_ids]                   # [n, 128]
    position_embeddings = model.position_embedding_table[:number_of_letters]     # [n, 128]
    return token_embeddings + position_embeddings                                # [n, 128]


example_ids = letters_to_ids("goo")
print("IDs for 'goo':", example_ids)
example_start = first_hidden_states(example_ids, model)
print("shape of the first hidden states:", list(example_start.shape))
print("the two o's start differently, because their positions differ:")
print("  position 2:", example_start[1][:3])
print("  position 3:", example_start[2][:3])

# %% [markdown]
# ## 6. Attention: each letter looks back at the letters before it
#
# This is the only place in the model where one letter's hidden state is affected by another's.
#
# For every position, attention makes three vectors from its (normalised) hidden state, each with its
# own table of weights. They are usually called the **query**, the **key**, and the **value**. Those
# names suggest a meaning that nobody has shown the numbers have, so I add a word to each that says
# only what it does in the arithmetic:
#
# - the **looking query** is used when this position looks back at the others;
# - the **looked-at key** is used when another position looks at this one;
# - the **passed-on value** is what this position passes on to whoever looks at it.
#
# Attention runs four times side by side, each time on its own 32 of the 128 numbers. Each of these is
# usually called a **head**; here it is a **head slice**, because that is all it is: a slice of the
# columns. Then, for each head slice separately:
#
# 1. Each position's looking query is compared with the looked-at key of every position up to and
#    including itself, by a dot product. A big result counts as a good match.
# 2. The match scores are divided by √32, the square root of the numbers per head slice. The 2017 authors'
#    reason is a suspicion, in their words: "We suspect that for large values of d_k, the dot products
#    grow large in magnitude, pushing the softmax function into regions where it has extremely small
#    gradients" (https://arxiv.org/abs/1706.03762, section 3.2.1).
# 3. Scores for later positions are set to minus infinity, so softmax gives them a share of exactly 0.
#    A letter may not look ahead, because the next letter is what it is trying to guess.
# 4. Softmax turns each position's scores into shares that add up to 1.
# 5. Each position collects the passed-on values of the earlier positions, weighted by those shares.
#
# Finally, the four head slices' 32-number findings are laid side by side (128 numbers again) and mixed by
# one more table of weights.
#
# **What we do not know.** The names "query", "key", and "value" come from database lookup, and they
# are an analogy chosen by the 2017 authors, not a description of what the numbers mean. The arithmetic
# above is all that happens. Training makes the query, key, and value weights useful, but *why* this
# particular design learns so well is not well understood (see "Open Problems in Mechanistic
# Interpretability", https://arxiv.org/abs/2501.16496). A few heads have a habit you can measure:
# in this model, block 1's first head mostly looks one position back. Most heads, in most models, have
# no tidy description at all, and the meaning people give to queries and keys is usually a story told
# afterwards. Even attention shares are not a safe guide to what matters: "very different attention
# distributions" can "yield equivalent predictions" (https://arxiv.org/abs/1902.10186).
#
# `torch.ones(n, n, dtype=torch.bool)` makes an *n* × *n* table filled with True, and `torch.tril`
# keeps the lower-left triangle, including the diagonal, and sets the rest to False: row *i* is True
# for columns 0 to *i*. `table[:, first:last]` keeps columns `first` to `last - 1` of every row.
# `.masked_fill(condition, value)` replaces every number where the condition is True. `torch.cat`
# joins tables side by side.

# %%
def attention(hidden_states: torch.Tensor, block: TrainedBlock) -> torch.Tensor:
    """[n, 128] -> [n, 128]: what attention adds to each hidden state."""
    number_of_positions = hidden_states.shape[0]
    normalised = normalise(hidden_states, block.stretch_before_attention, block.shift_before_attention)

    looking_queries = apply_weights(normalised, block.looking_query_weights, block.looking_query_biases)      # [n, 128]
    looked_at_keys = apply_weights(normalised, block.looked_at_key_weights, block.looked_at_key_biases)      # [n, 128]
    passed_on_values = apply_weights(normalised, block.passed_on_value_weights, block.passed_on_value_biases)  # [n, 128]

    # may_look_at[i][j] is True when position i may look at position j: only j <= i.
    may_look_at = torch.tril(torch.ones(number_of_positions, number_of_positions, dtype=torch.bool))

    findings_of_each_head_slice = []
    for head_slice in range(HEADS_PER_BLOCK):
        first_column = head_slice * NUMBERS_PER_HEAD
        after_last_column = first_column + NUMBERS_PER_HEAD
        head_looking_queries = looking_queries[:, first_column:after_last_column]    # [n, 32]
        head_looked_at_keys = looked_at_keys[:, first_column:after_last_column]      # [n, 32]
        head_passed_on_values = passed_on_values[:, first_column:after_last_column]  # [n, 32]

        # match_scores[i][j]: position i's looking query · position j's looked-at key
        match_scores = head_looking_queries @ head_looked_at_keys.T          # [n, n]
        shrunk_scores = match_scores / math.sqrt(NUMBERS_PER_HEAD)          # [n, n]
        scores_without_later_positions = shrunk_scores.masked_fill(~may_look_at, float("-inf"))
        attention_shares = scores_to_shares(scores_without_later_positions)  # [n, n], rows add up to 1

        # Each position collects every position's passed-on values, weighted by its shares of attention.
        what_this_head_slice_collected = attention_shares @ head_passed_on_values   # [n, 32]
        findings_of_each_head_slice.append(what_this_head_slice_collected)

    all_head_slices_side_by_side = torch.cat(findings_of_each_head_slice, dim=-1)   # [n, 128]
    return apply_weights(all_head_slices_side_by_side, block.head_mixing_weights, block.head_mixing_biases)


# %% [markdown]
# The same arithmetic, one head slice, with the real numbers for `goo`: position 3's shares of attention
# in block 1, head slice 1. (These are the numbers in Part 1's attention section: 4.0%, 92.9%, and 3.1%.)

# %%
def attention_shares_for_one_head_slice(hidden_states, block, head_slice):
    """The [n, n] table of shares for one head slice, worked out exactly as in attention() above."""
    n = hidden_states.shape[0]
    normalised = normalise(hidden_states, block.stretch_before_attention, block.shift_before_attention)
    columns = slice(head_slice * NUMBERS_PER_HEAD, (head_slice + 1) * NUMBERS_PER_HEAD)
    head_looking_queries = apply_weights(normalised, block.looking_query_weights, block.looking_query_biases)[:, columns]
    head_looked_at_keys = apply_weights(normalised, block.looked_at_key_weights, block.looked_at_key_biases)[:, columns]
    may_look_at = torch.tril(torch.ones(n, n, dtype=torch.bool))
    shrunk_scores = head_looking_queries @ head_looked_at_keys.T / math.sqrt(NUMBERS_PER_HEAD)
    return scores_to_shares(shrunk_scores.masked_fill(~may_look_at, float("-inf")))


shares = attention_shares_for_one_head_slice(example_start, model.blocks[0], head_slice=0)
print("position 3's shares of attention, block 1, head slice 1:", [f"{s:.1%}" for s in shares[2].tolist()])

# %% [markdown]
# ## 7. The MLP: each letter on its own
#
# The MLP ("multilayer perceptron") works on each position's hidden state separately: no position looks
# at any other. It normalises, widens the 128 numbers to 512 with one table of weights, bends them, and
# narrows them back to 128 with a second table.
#
# What is it for? A common rule of thumb is that attention brings in *context* from other letters, and
# the MLP brings in *knowledge* stored in its weights, such as which letter usually ends a word. There
# is evidence for that in big models (https://arxiv.org/abs/2012.14913), but it is a rough picture, not
# a proven account (https://arxiv.org/abs/2301.04213), and in a model
# this small nobody has worked out what each of the 512 numbers does. Why 512, four times 128? A
# convention from the 2017 design, kept because it works, not derived from anything.

# %%
def mlp(hidden_states: torch.Tensor, block: TrainedBlock) -> torch.Tensor:
    """[n, 128] -> [n, 128]: what the MLP adds to each hidden state."""
    normalised = normalise(hidden_states, block.stretch_before_mlp, block.shift_before_mlp)
    widened = apply_weights(normalised, block.widen_weights, block.widen_biases)       # [n, 512]
    bent = bend(widened)                                                               # [n, 512]
    return apply_weights(bent, block.narrow_weights, block.narrow_biases)              # [n, 128]


# %% [markdown]
# ## 8. One block, then four
#
# A block runs attention, then the MLP, and **adds** what each one produces to the hidden states rather
# than replacing them. The original code writes this as `x = x + self.attn(self.ln1(x))`, which reuses
# the name `x` for both the old and the new hidden states. Here, the old and new states have different
# names, so you can see that nothing is thrown away: each block only adds to what was there.
#
# The four blocks run in order, each starting from the previous block's result. I keep every block's
# result in a list, so that the whole history of the hidden states can be inspected afterwards.

# %%
def run_one_block(hidden_states_entering: torch.Tensor, block: TrainedBlock) -> torch.Tensor:
    """[n, 128] -> [n, 128]"""
    attention_contribution = attention(hidden_states_entering, block)
    hidden_states_after_attention = hidden_states_entering + attention_contribution

    mlp_contribution = mlp(hidden_states_after_attention, block)
    hidden_states_leaving = hidden_states_after_attention + mlp_contribution
    return hidden_states_leaving


def run_all_blocks(starting_hidden_states: torch.Tensor, model: TrainedModel) -> list:
    """Returns the hidden states before block 1 and after each block: a list of 5 tables, each [n, 128]."""
    hidden_states_after_each_block = [starting_hidden_states]
    for block in model.blocks:
        hidden_states_entering = hidden_states_after_each_block[-1]   # the latest result
        hidden_states_after_each_block.append(run_one_block(hidden_states_entering, block))
    return hidden_states_after_each_block


history = run_all_blocks(example_start, model)
for stage, hidden_states in enumerate(history):
    label = "before block 1" if stage == 0 else f"after block {stage}"
    size = hidden_states[2].norm().item()   # .norm(): the length of the vector, √(sum of squares)
    print(f"{label:15} position 3's hidden state has size {size:.2f}")

# %% [markdown]
# ## 9. From the last hidden state to 65 scores
#
# Only the last letter's hidden state is needed to guess the next letter. (The original code scores
# every position and then keeps only the last row, because training uses all of them. The answer is the
# same.) The last hidden state is normalised one final time, giving the **final hidden state**, and then
# compared with each of the 65 next-letter rows by a dot product, plus that row's bias. The result is 65
# scores, the **logits**: one for each letter that could come next.

# %%
def scores_for_next_letter(text: str, model: TrainedModel) -> torch.Tensor:
    """A text -> [65] scores, one for each letter that could come next."""
    letter_ids = letters_to_ids(text)[-MOST_LETTERS_THE_MODEL_CAN_SEE:]   # only the last 128 letters fit
    starting_hidden_states = first_hidden_states(letter_ids, model)
    hidden_states_after_each_block = run_all_blocks(starting_hidden_states, model)
    last_letters_hidden_state = hidden_states_after_each_block[-1][-1]   # last block, last position: [128]
    final_hidden_state = normalise(last_letters_hidden_state, model.final_stretch, model.final_shift)
    return model.next_letter_rows @ final_hidden_state + model.next_letter_biases   # [65]


scores = scores_for_next_letter("go", model)
for letter in "o d":
    print(f"score for {letter!r} after 'go': {scores[LETTER_TO_ID[letter]]:.3f}")

# %% [markdown]
# ### Checking against the original
#
# The point of this rewrite is readability, so it must not change the answers. This cell builds the
# original notebook's model (its class `MiniGPT`, from Jibin Joseph's code, which the series repo
# downloads), loads the same trained numbers, and compares the scores for several texts. The biggest
# difference should be a rounding difference, far below 0.001.

# %%
try:
    import sys
    sys.path.insert(0, str(Path.cwd().parent))   # the series repo, when this runs from readable/
    if not (Path.cwd().parent / "minigpt_notebook.py").exists():   # in Colab, which has only this notebook
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Haddley/minigpt-series/main/minigpt_notebook.py", "minigpt_notebook.py")
        sys.path.insert(0, str(Path.cwd()))
    from minigpt_notebook import load_model_classes   # downloads and runs the original model code
    original_namespace = {}
    load_model_classes(original_namespace)
    original_model = original_namespace["MiniGPT"](original_namespace["GPTConfig"]())
    original_model.load_state_dict(stored_tables)
    original_model.eval()   # switches off dropout, which is only used in training
    biggest_difference = 0.0
    for text in ["go", "goo", "ROMEO:\nI thoug", "First Citizen:\nBefore we proceed any further, hear me spea"]:
        with torch.no_grad():   # no_grad: do not record the steps, which only training needs
            original_logits, _ = original_model(torch.tensor([letters_to_ids(text)]))
        original_scores = original_logits[0, -1]
        readable_scores = scores_for_next_letter(text, model)
        biggest_difference = max(biggest_difference, (original_scores - readable_scores).abs().max().item())
    print(f"biggest difference from the original, over 4 texts and 65 scores each: {biggest_difference:.2e}")
except Exception as problem:   # the check needs the series repo; the rest of the program does not
    print("could not load the original code for the check:", problem)

# %% [markdown]
# ## 10. From scores to the next letter
#
# Two settings reshape the 65 scores before the wheel is spun. They are chosen each time the model
# writes; they are not learned.
#
# - **Temperature** divides every score by one number. Below 1 it stretches the gaps between scores, so
#   the favourite gets more of the wheel; above 1 it shrinks them, so the slices even out.
# - **Top-k** keeps only the *k* biggest scores and sets the rest to minus infinity, so that softmax
#   gives them a chance of exactly 0. `torch.topk(scores, k)` returns the *k* biggest numbers, biggest
#   first; `.values[-1]` is the smallest of them.
#
# Then softmax turns the scores into 65 chances that add up to 1: the wheel.
#
# Spinning the wheel picks a random point between 0 and 1 and walks along the chances, adding them up,
# until the running total passes that point. `torch.cumsum` makes the running totals, and
# `torch.searchsorted` finds where the random point falls among them. A random-number generator with a
# fixed seed gives the same "random" spins every time, so the results can be repeated.

# %%
def chances_from_scores(scores: torch.Tensor, temperature: float, keep_biggest: int) -> torch.Tensor:
    """[65] scores -> [65] chances that add up to 1."""
    if temperature <= 0:   # temperature 0 means "always the biggest score": give it the whole wheel
        whole_wheel_to_the_favourite = torch.zeros_like(scores)
        whole_wheel_to_the_favourite[scores.argmax()] = 1.0
        return whole_wheel_to_the_favourite
    scaled_scores = scores / temperature
    smallest_score_kept = torch.topk(scaled_scores, min(keep_biggest, len(scaled_scores))).values[-1]
    trimmed_scores = scaled_scores.masked_fill(scaled_scores < smallest_score_kept, float("-inf"))
    return scores_to_shares(trimmed_scores)


def spin_the_wheel(chances: torch.Tensor, random_generator: torch.Generator) -> int:
    """Pick one letter ID, each with its own chance."""
    random_point = torch.rand(1, generator=random_generator)     # one number between 0 and 1
    running_totals = torch.cumsum(chances, dim=0)                # [65], ends at 1
    landed_on = torch.searchsorted(running_totals, random_point).item()
    return min(landed_on, len(chances) - 1)                      # a safety net for rounding


def choose_next_letter(text: str, model: TrainedModel, temperature: float, keep_biggest: int,
                       random_generator: torch.Generator) -> str:
    """The whole model, top to bottom: one line for each step in the post's whole-program picture."""
    letter_ids = letters_to_ids(text)[-MOST_LETTERS_THE_MODEL_CAN_SEE:]                                 # step 1
    starting_hidden_states = first_hidden_states(letter_ids, model)                                     # step 2
    hidden_states_after_each_block = run_all_blocks(starting_hidden_states, model)                     # step 3
    last_letters_hidden_state = hidden_states_after_each_block[-1][-1]                                 # step 4
    final_hidden_state = normalise(last_letters_hidden_state, model.final_stretch, model.final_shift)  # step 5
    scores = model.next_letter_rows @ final_hidden_state + model.next_letter_biases                    # step 6
    chances = chances_from_scores(scores, temperature, keep_biggest)                                   # step 7
    next_letter_id = spin_the_wheel(chances, random_generator)                                         # step 8
    return VOCABULARY[next_letter_id]


def write(start: str, letters_to_add: int, model: TrainedModel,
          temperature: float = 0.8, keep_biggest: int = 65, seed: int = 0) -> str:
    random_generator = torch.Generator().manual_seed(seed)
    text_so_far = start
    for _ in range(letters_to_add):
        next_letter = choose_next_letter(text_so_far, model, temperature, keep_biggest, random_generator)
        text_so_far = text_so_far + next_letter   # the only name that changes: the text itself grows
    return text_so_far


chances = chances_from_scores(scores_for_next_letter("go", model), temperature=1.0, keep_biggest=65)
print("chance of 'o' after 'go':", f"{chances[LETTER_TO_ID['o']]:.1%}")
print()
print(write("ROMEO:\n", 200, model, temperature=0.8, seed=1))

# %% [markdown]
# `choose_next_letter` is the whole model on one screen: steps 1 to 6 are the same lines as
# `scores_for_next_letter` in section 9, and steps 7 and 8 choose the letter. Every other function in
# this program is one of the pieces it calls.
#
# `text_so_far = text_so_far + next_letter` is the one place where a name is reused for a new value.
# I left it, because "the text so far" really is one thing that grows letter by letter, and the loop
# would be harder to follow with a new name for every length.
#
# That is the whole model: four small tools, attention, the MLP, four blocks, a last dot product, and a
# wheel. Everything it knows is in the 826,433 numbers of section 2.

# %% [markdown]
# ## Try it yourself
#
# Change the starting text, and run this cell again. Try a different `seed` for a different spin of the
# wheel, a lower `temperature` (such as 0.3) for safer guesses, or a higher one (such as 1.5) for wilder
# ones. The model only knows its 65 letters: `letters_to_ids` quietly skips any other letter, such as
# `é` or `7`.

# %%
print(write("KING RICHARD III:\nA horse! a horse! my kingdom for a hors", 200, model, temperature=0.8, seed=1))
