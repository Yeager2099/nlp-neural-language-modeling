# Report transcription

Text extracted from the author-provided course report. Figures are not reproduced and tables may lose formatting. Student identifiers and institutional email addresses are removed.

## Page 1

CIS 5300: Natural Language Processing
Homework 5: Neural Architectures (RNN →
Transformer)
Report Template
Name: Yeager Ye
Penn ID: [STUDENT_ID_REDACTED]
Learning Objectives
1. Implement character-level tokenization with encode/decode functions that
convert between text and integer sequences.
2. Build an RNN language model using PyTorch's `nn.RNN`, understanding the
recurrent hidden state as a simple sequence model baseline.
3. Construct a causal attention mask that prevents tokens from attending to future
positions, enabling autoregressive generation.
4. Implement scaled dot-product attention with query, key, and value projections,
understanding why the scaling factor is necessary.
5. Build multi-head attention by running multiple attention heads in parallel and
concatenating their outputs.
6. Assemble a transformer block with attention, feedforward layers, residual
connections, and layer normalization.
7. Train a complete transformer language model on the Tiny Shakespeare dataset
and generate text autoregressively.
8. Compare RNN and Transformer architectures through experimental results and
written analysis.

## Page 2

CIS 5300: Natural Language Processing
Points Breakdown
Section Points Description
Answer 4.4 3 pts Generation Failure Analysis
Answer 5.1 5 pts Attention Patterns
Answer 5.2 5 pts From RNNs to Transformers
Answer 6.1 7 pts Why Scaled Dot-Product
Total 20 pts
Answer 4.4: Generation Failure Analysis
Generate at least 300 characters of new output with temperature=1.0 and identify 3
specific failures from the categories below. For each, quote the relevant passage
and explain why your model produces this error.
1. Repetition: Does the model get stuck repeating a phrase or pattern? Why might a
small model with limited context do this?
2. Nonsense words: Does it generate character sequences that aren't real English
words? Why does character-level tokenization make this more likely than word-
level?
3. Grammatical errors: Does it start a sentence or speech prefix correctly but lose
coherence? Connect to the model's limited context window (`block_size=64`
characters ~10–12 words).
4. Inconsistent character names: Does it mix up Shakespeare characters or invent
new ones? Why?

## Page 3

CIS 5300: Natural Language Processing
Paste at least 300 characters of generated output, then identify and analyze 3
specific failures. Quote the relevant passage for each failure and explain the
underlying cause. Connect to the model architecture — e.g., context window size,
character-level tokenization, model capacity.
Below is a 300+ character generation sample from the trained Transformer (temperature=1.0,
prompt: "ROMEO:\nWherefore art thou?"):
ROMEO: Wherefore art thou? CARENTIUS: Stame? LONTAD: Cul say cont not rect
booy, have art his heaveth your stail Waste hight be the dward'st heeave; and
though, that a pomtage: And will speal be that on, I'll are hand to will have
I way to mines they his im! FORY: Davy, be it andothin wortwerely with oness
as a your! In t
Failure 1: Nonsense Words
The model generates many non-existent English words such as "CARENTIUS," "Stame,"
"LONTAD," "booy," "pomtage," "speal," "andothin," and "wortwerely." Character-level tokenization
makes this far more likely than word-level tokenization because the model must predict individual
characters rather than complete vocabulary words. With only 65 possible characters, the model
can easily combine them into plausible-looking but meaningless sequences. A word-level model
would be constrained to output actual dictionary words, though it would struggle with out-of-
vocabulary terms. The small model capacity (816K parameters) and limited training data further
prevent the model from learning robust English spelling patterns.
Failure 2: Inconsistent Character Names
The model invents entirely new character names like "CARENTIUS," "LONTAD," and "FORY"
that do not exist in Shakespeare's works. This happens because the model has learned the
pattern of capitalized names followed by colons (a common dialogue format in Shakespeare), but
with only a 64-character context window, it cannot reliably remember which characters have
already appeared in the scene. The model generates names that follow the orthographic pattern
of Shakespearean names (multi-syllabic, Latin-sounding) but are essentially hallucinations. A
larger model with longer context or word-level tokenization would have a better chance of
maintaining character consistency.
Failure 3: Grammatical Errors and Loss of Coherence
The model begins with a coherent Shakespearean dialogue format but rapidly loses syntactic
structure. For example, "Cul say cont not rect booy, have art his heaveth your stail" has no
discernible grammatical structure. The limited context window of 64 characters (approximately
10-12 words) means the model can only condition on a very short prefix when predicting the next
token. Complex grammatical dependencies in English often span more than 10-12 words, so the
model cannot maintain subject-verb agreement, proper clause structure, or semantic coherence

## Page 4

CIS 5300: Natural Language Processing
across longer passages. Each prediction is made with severely truncated context, leading to
locally plausible but globally incoherent output.

## Page 5

CIS 5300: Natural Language Processing
Answer 5.1: Attention Patterns
After training your transformer, extract and visualize the attention weights from at
least 4 different heads (across at least 2 layers) using the code cell in Section 5.
a. Describe what patterns you observe across the heads. Look for:
i. Local/positional attention: Does any head consistently attend to the
immediately preceding token?
ii. Content-based attention: Does any head attend to specific characters
(spaces, newlines, punctuation) regardless of position?
iii. Diffuse attention: Does any head spread attention roughly uniformly?
b. Do different heads within the same layer show different patterns? What about
the same head position across different layers?
c. Do you see any evidence of "induction heads" — a pattern where the model
attends back to the previous occurrence of a token to predict what came after it?
Include your attention heatmaps (at least 4 heads across 2+ layers). For each head,
briefly describe the pattern you observe. Respond to parts (a), (b), and (c) with
reference to specific heads and layers. Use the character labels on the axes to
identify any content-based patterns.

## Page 6

CIS 5300: Natural Language Processing
After training the Transformer, I extracted and visualized attention weights from 4 different
heads across 3 layers using the sample input "To be, or not to be,". The heatmaps are shown
below.
Figure 1: Layer 0, Head 0
Figure 2: Layer 0, Head 2
Figure 3: Layer 2, Head 0

## Page 7

CIS 5300: Natural Language Processing
Figure 4: Layer 3, Head 1
a. Pattern Descriptions
Positional/Local Attention: Layer 0, Head 0 (Figure 1) shows a strong diagonal pattern where
each token attends most heavily to itself and the immediately preceding tokens. This is visible
as a bright diagonal line running from top-left to bottom-right. This pattern likely serves a basic
positional tracking function, allowing the model to maintain local coherence by referencing
nearby context when predicting the next character.
Content-Based Attention: Layer 0, Head 2 (Figure 2) shows vertical stripes at positions
corresponding to spaces and commas, indicating this head attends to whitespace and
punctuation characters regardless of their position. This suggests the head has learned to use
structural characters as anchors for parsing the text stream, which is useful for identifying word
boundaries in character-level tokenization.
Diffuse Attention: Layer 2, Head 0 (Figure 3) shows a more diffuse, broadly distributed
attention pattern with relatively uniform weights across all previous positions. This type of head
may serve an aggregation or averaging function, collecting general contextual information from
the entire available prefix rather than focusing on specific positions. Such diffuse attention can
help the model maintain a broad sense of topic or register across the sequence.
b. Cross-Head and Cross-Layer Comparisons
Within the same layer, different heads show distinctly different patterns. In Layer 0, Head 0
exhibits strong local/diagonal attention while Head 2 shows content-based attention to
punctuation. This confirms that multi-head attention successfully learns complementary
functions, with different heads specializing in different types of relationships. Across layers, the
same head position shows evolution: Layer 3, Head 1 (Figure 4) displays a sharper, more
selective pattern compared to the more diffuse attention seen in earlier layers. Deeper layers
appear to develop more refined, task-specific attention patterns, while earlier layers handle
more basic positional and structural features.

## Page 8

CIS 5300: Natural Language Processing
c. Evidence of Induction Heads
The sample "To be, or not to be," contains a repeated substring ("be" appears twice). In Layer
0, Head 2, I observe some increased attention weight from the second occurrence of characters
back to their first occurrence, which is consistent with the induction head pattern described by
Olsson et al. (2022). However, the evidence is not as clear-cut as in larger models because our
model is small (4 layers, 4 heads, 128-dimensional embeddings) and trained on limited data.
Induction heads typically emerge more clearly in models with 6+ layers and larger embedding
dimensions. The character-level tokenization also makes induction patterns harder to interpret
since repeated substrings are shorter and more common.

## Page 9

CIS 5300: Natural Language Processing
Answer 5.2: From RNNs to Transformers
You trained an RNN baseline (Section 1.3) and a Transformer (Section 4) on the
same Tiny Shakespeare next-character prediction task.
a. Validation loss. Report the final validation loss of your RNN baseline (after
1,000 iterations) and your Transformer (after full training). Which performs
better, and by how much?
b. Architectural reasons. RNNs have two well-known limitations: sequential
processing and the long-range information bottleneck. How does the
Transformer address each? Be specific about which Transformer components
(e.g., self-attention, position embeddings, parallel computation) map to which
RNN limitation.
c. Tradeoffs. Are there any aspects in which RNNs still have an advantage over
Transformers? Think about memory, parameter count, or sequence-length
scaling.
Report specific validation loss numbers for both models. Explain the architectural
differences in your own words. Hint: for (c), consider inference-time memory (KV
cache vs. fixed hidden state) and the cost of attention for RNNs.]
a. Validation Loss Comparison
RNN Baseline (1000 iterations): Final validation loss = 2.0538
Transformer (2500 iterations): Final validation loss = 1.9013
The Transformer outperforms the RNN baseline by approximately 0.15 nats in cross-entropy loss
(a relative improvement of about 7.4%). This is a meaningful difference given that both models
were trained on identical data with similar hyperparameters. The lower validation loss indicates
that the Transformer generalizes better to unseen Shakespeare text.
b. Architectural Reasons
Sequential Processing Limitation: RNNs process sequences one time step at a time because
the hidden state at each step depends on the hidden state from the previous step. This sequential
dependency prevents parallelization during training. The Transformer addresses this through self-
attention, which computes relationships between all pairs of tokens simultaneously. In our
implementation, the attention scores QK^T are computed in a single matrix multiplication, allowing

## Page 10

CIS 5300: Natural Language Processing
the entire sequence to be processed in parallel. Position embeddings (self.pos_emb) provide the
sequential information that would otherwise come from the recurrent state.
Long-Range Information Bottleneck: In RNNs, information from early time steps must pass
through each intermediate hidden state, where it can be diluted or overwritten by newer inputs.
This creates a bottleneck for long-range dependencies. The Transformer eliminates this
bottleneck through direct attention connections: any token can attend directly to any previous
token, regardless of distance. In our model, the first token in a 64-character context window can
directly influence the prediction of the 64th token through the attention mechanism, with no
intermediate compression. The residual connections (x = x + attn(ln1(x))) further preserve
information across layers by allowing gradients and signals to flow directly through the network.
c. Tradeoffs
Despite their superior performance on this task, RNNs still offer advantages over Transformers in
certain scenarios:
1. function bold() { [native code] }RNNs maintain a fixed-size hidden state vector during inference,
requiring O(1) memory per time step regardless of sequence length. Transformers, by contrast,
require storing the key and value tensors for all previous positions (the KV cache), which grows
linearly with sequence length (O(T) memory). For very long sequences or memory-constrained
deployment, RNNs can be more efficient at inference time.
2. function bold() { [native code] }The self-attention mechanism in Transformers has O(T^2) time
and memory complexity with respect to sequence length because every token attends to every
other token. For long sequences, this quadratic scaling becomes prohibitive. RNNs have O(T)
complexity per sequence, making them more scalable for very long sequences (though with
degraded quality for long-range dependencies).
3. function bold() { [native code] }For the same effective capacity, RNNs typically have fewer
parameters because they reuse the same recurrent weights across all time steps. Our
Transformer has 816,705 parameters, while an RNN with comparable embedding dimensions
would have significantly fewer. This makes RNNs attractive for resource-constrained
environments where model size must be minimized.

## Page 11

CIS 5300: Natural Language Processing
Answer 6.1: Why Scaled Dot-Product
The attention formula divides the dot product 𝑄𝐾! by !𝑑 :
𝑘
𝑄𝐾!
𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄,𝐾,𝑉) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥( )𝑉
5𝑑
#
a. What would happen if you removed this scaling factor? What happens to the
softmax output as 𝑑 grows?
#
b. Why does the magnitude of 𝑄𝐾! grow with 𝑑 ? Assume the entries of 𝑄 and 𝐾
#
are independent random variables with mean 0 and variance 1. What is the
variance of a single element of
𝑄𝐾!
?
c. Why does dividing by !𝑑 fix this problem? What is the variance of the scaled
𝑘
dot product?
For (a), 'think about what softmax does when its inputs have very large magnitude'.
For (b), show the variance calculation: each element of 𝑄𝐾! is a dot product of 𝑑
#
independent terms. For (c), apply the scaling rule: 𝑉𝑎𝑟(𝑋/𝑐) = 𝑉𝑎𝑟(𝑋)/𝑐2 .

## Page 12

CIS 5300: Natural Language Processing

## Page 13

CIS 5300: Natural Language Processing
