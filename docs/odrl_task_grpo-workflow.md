# Using domain datasets and ODRL tasks with GRPO

## 1. Main idea

GRPO is not limited to Countdown arithmetic. It can be used for mathematics,
code, SQL, contracts, legal reasoning, ODRL policies, or another specialist
domain.

The important requirement is a reliable reward function. A useful GRPO record
therefore needs more than raw text:

```text
problem or prompt
    + expected facts, structure, or outcome
    + an automatic verification procedure
```

Raw contracts can be useful for continued pretraining, but raw contracts alone
are not yet a GRPO dataset. They must be converted into tasks whose generated
answers can be scored.

## 2. Examples of suitable domains

| Domain | Example data | Possible automatic verifier |
|---|---|---|
| Arithmetic | Countdown, GSM8K, MATH | Numeric or symbolic equivalence |
| Programming | HumanEval, MBPP, APPS, CodeContests | Compile and execute unit tests |
| SQL | Spider, BIRD | Execute the query and compare results |
| Formal proof | Lean theorem datasets, miniF2F | Lean proof checker |
| Logic and puzzles | SAT, Sudoku, scheduling, graph problems | Deterministic solver |
| Contract review | CUAD | Clause category and evidence-span matching |
| Contract inference | ContractNLI | Exact NLI label and evidence matching |
| Legal reasoning | Selected LegalBench tasks | Exact label or structured result |
| Rights policies | W3C ODRL examples and custom policies | JSON-LD, RDF, vocabulary, and policy evaluation |

Useful public references include:

- [GSM8K](https://github.com/openai/grade-school-math)
- [MATH](https://github.com/hendrycks/math)
- [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [CUAD](https://huggingface.co/datasets/theatticusproject/cuad)
- [ContractNLI](https://github.com/stanfordnlp/contract-nli)
- [LegalBench](https://github.com/HazyResearch/legalbench)

Benchmark test records should be kept separate from training records. Training
on the complete benchmark would make the final benchmark result unreliable.

## 3. Contract tasks suitable for GRPO

### 3.1 Clause extraction

Given a contract and a clause category, ask the model to identify the relevant
text. CUAD contains contract passages annotated with important clause types.

A structured response could be:

```json
{
  "clause_type": "Termination For Convenience",
  "evidence_start": 1482,
  "evidence_end": 1795
}
```

The verifier can check:

1. whether the response is valid JSON;
2. whether the category belongs to the allowed set;
3. whether the predicted span overlaps the annotated span;
4. whether the quoted evidence is actually present in the source contract.

### 3.2 Contract natural-language inference

ContractNLI asks whether a hypothesis is:

- `Entailment`;
- `Contradiction`; or
- `NotMentioned`.

It also contains evidence annotations. A model response could therefore be:

```json
{
  "decision": "Entailment",
  "evidence_sentence_ids": [14, 15]
}
```

An illustrative reward is:

$$
R
=
1.0R_{\text{decision}}
+
0.5R_{\text{evidence}}
+
0.1R_{\text{format}}.
$$

Here, the decision reward uses exact match, the evidence reward can use
sentence-level F1, and the format reward checks the JSON schema. The main task
reward should dominate the formatting reward.

## 4. Why ODRL is promising for GRPO

The [W3C ODRL Information Model](https://www.w3.org/TR/odrl-model/) defines
structured concepts such as:

- policies;
- permissions;
- prohibitions;
- duties and consequences;
- assigners and assignees;
- target assets and actions;
- constraints;
- policy inheritance and conflict strategies.

The [W3C ODRL Vocabulary and Expression
2.2](https://www.w3.org/TR/odrl-vocab/) defines standard terms and JSON-LD
representations. This structure makes many ODRL outputs mechanically
checkable, which is valuable for rule-based reinforcement learning.

There is not one canonical, large ODRL reasoning dataset comparable to GSM8K.
A practical project will probably combine W3C examples, programmatically
generated policies, manually reviewed cases, and permitted private domain
data.

## 5. Possible ODRL GRPO tasks

### Task A: natural language to ODRL

Input:

```text
Alice permits Bob to display asset A until 31 December 2027, provided Bob
attributes Alice.
```

Output: an ODRL JSON-LD policy containing the permission, parties, asset,
action, time constraint, and attribution duty.

### Task B: ODRL policy interpretation

Given an ODRL policy and a proposed action, return:

```json
{
  "decision": "conditional_permission",
  "required_duties": ["attribute"],
  "supporting_rule_ids": ["permission-1"]
}
```

### Task C: policy conflict detection

Give the model permissions, prohibitions, and a conflict strategy. Ask it to
identify conflicts and calculate the resulting decision.

### Task D: policy repair

Provide an invalid or incomplete ODRL policy and ask the model to return the
smallest valid correction.

### Task E: contract clause to ODRL

Provide a real or synthetic contract clause and ask the model to encode its
rights, restrictions, parties, assets, constraints, and duties as ODRL.

## 6. Recommended ODRL record format

An internal training record could look like:

```json
{
  "id": "odrl-000001",
  "task_type": "policy_interpretation",
  "prompt": "...policy and proposed use...",
  "expected_decision": "prohibited",
  "expected_rule_ids": ["prohibition-2"],
  "expected_policy_graph": "...canonical RDF or JSON-LD...",
  "profile": "http://www.w3.org/ns/odrl/2/core",
  "difficulty": 2,
  "source": "synthetic-reviewed"
}
```

Do not put a hidden answer or reference reasoning in the prompt. The expected
fields are used only by the reward verifier.

Useful metadata includes source, licence, jurisdiction, governing date,
document identifier, and train/evaluation split. Contract-level splitting is
important: clauses from the same contract should not appear in both training
and evaluation data.

## 7. A layered ODRL reward

A verifier can score progressively stronger properties:

```text
valid JSON
    -> valid JSON-LD or RDF
    -> allowed ODRL vocabulary and required fields
    -> graph agreement with the reference policy
    -> correct policy decision and supporting evidence
```

An illustrative reward is:

$$
R
=
0.05R_{\text{JSON}}
+
0.10R_{\text{schema}}
+
0.20R_{\text{ODRL}}
+
0.25R_{\text{graph}}
+
0.40R_{\text{decision}}.
$$

The values above are starting points, not universal optimal weights. The
semantic decision receives the greatest weight so that the model cannot obtain
a high reward merely by producing attractive, syntactically valid JSON.

### 7.1 Verification details

The verifier should:

1. parse JSON without executing generated code;
2. validate required keys and value types;
3. expand JSON-LD into an RDF graph;
4. reject unknown terms unless they belong to the selected ODRL profile;
5. compare normalized graphs rather than raw JSON strings;
6. run the proposed use through a deterministic policy evaluator;
7. verify that returned rule IDs exist in the input policy;
8. apply length and timeout limits to every parser and evaluator.

Raw JSON string equality is unsuitable because property ordering and valid
JSON-LD aliases can differ while expressing the same graph. Blank nodes also
require graph-isomorphism or canonicalization handling.

Schema or SHACL validity alone does not prove that a policy makes the correct
decision. Structural validation and semantic evaluation must be separate
reward components.

ODRL permits profiles and extensible vocabulary. For a deterministic training
task, define a closed supported profile and explicitly document its conflict,
constraint, and missing-information semantics.

## 8. Group construction and trajectory collection

The existing Countdown GRPO workflow can remain largely unchanged:

1. Select several different ODRL problems.
2. Sample multiple responses for each problem.
3. Treat the responses to one problem as one GRPO group.
4. Score each response with the ODRL verifier.
5. Standardize rewards inside that group.
6. Save old-policy token log probabilities.
7. Update the policy with the clipped GRPO objective.

For response $i$ within one group:

$$
\widehat A_i
=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}.
$$

If every response receives the same reward, the group supplies no relative
learning signal. Layered rewards and a curriculum can reduce the frequency of
zero-variance groups.

One practical curriculum is:

1. valid JSON generation;
2. small policies with one permission or prohibition;
3. duties and simple constraints;
4. multiple interacting rules;
5. conflicts, inheritance, and profile-specific vocabulary;
6. contract clauses with ambiguity and long context.

## 9. Adapting the current implementation

The current project separates domain logic from the GRPO algorithm. A new
ODRL experiment would mainly require the following replacements:

| Current component | ODRL replacement |
|---|---|
| `CountdownDataset` | `ODRLDataset` |
| `Problem.numbers` and `Problem.target` | Policy, proposed use, expected outcome, and metadata |
| Countdown prompt template | ODRL task template and explicit output schema |
| Arithmetic AST verifier | JSON-LD/RDF/ODRL verifier and policy evaluator |
| `answer_reward()` | `odrl_semantic_reward()` |
| Arithmetic accuracy metric | Decision accuracy, graph score, and evidence score |

The rollout, old log-probability collection, group advantage calculation,
clipped importance ratio, reference KL penalty, gradient accumulation, and
checkpoint logic do not fundamentally need to change.

It is better to build `odrl_task.py` beside `countdown_task.py` than to mix ODRL
rules into the arithmetic verifier.

## 10. Strict R1-Zero versus practical domain post-training

A strict R1-Zero-style experiment starts from Qwen2.5-3B Base and performs
rule-reward RL without a reasoning-SFT warm-up:

```text
Qwen2.5-3B Base
    -> ODRL rule-reward GRPO
```

This is scientifically interesting, but its early rewards may be extremely
sparse. A base model might rarely generate valid ODRL or reliable legal
decisions, leaving many groups with identical zero rewards.

A more practical domain workflow is:

```text
Qwen2.5-3B Base
    -> continued pretraining on permitted domain text
    -> SFT on valid instruction-response ODRL examples
    -> GRPO with deterministic ODRL rewards
```

The second workflow is R1-style domain post-training, but it should not be
called strict R1-Zero because it contains domain pretraining or SFT before RL.

A useful compromise is to run both as controlled experiments:

- **Zero run:** Base model directly to ODRL GRPO;
- **SFT + GRPO run:** the same base checkpoint, data split, verifier, and GRPO
  settings, but with an ODRL SFT warm-up.

Compare valid-output rate, reward, zero-variance group rate, held-out semantic
accuracy, and generalization to unseen policy templates.

## 11. Domain-data cautions

For legal and contract data:

- verify dataset and document licences before training;
- remove confidential information and personal data;
- keep contracts, not merely individual clauses, separated across splits;
- record jurisdiction and governing date;
- test for memorization and template leakage;
- use qualified domain review for the held-out evaluation set;
- do not treat syntactic ODRL validity as proof of legal correctness;
- do not present model output as legal advice without appropriate professional
  review and system-level safeguards.

An LLM-as-judge reward can supplement human evaluation, but it is weaker than
a deterministic verifier and can introduce bias, inconsistency, and reward
hacking. Whenever possible, use exact structured outcomes and evidence tied
directly to the source document.

## 12. Recommended project plan

1. Keep Countdown as a test that the GRPO implementation is functioning.
2. Define a small, closed ODRL profile and deterministic policy evaluator.
3. Generate easy synthetic policies and proposed-use decisions.
4. Manually review a representative subset.
5. Reserve entire policy templates and real documents for evaluation.
6. Implement `odrl_task.py` and unit-test every reward component.
7. Run a short Qwen2.5-3B Base Zero experiment.
8. Measure valid JSON, valid ODRL, decision accuracy, and zero-variance groups.
9. If rewards remain too sparse, add curriculum stages or an ODRL SFT warm-up.
10. Only then scale the dataset, response length, and number of GRPO steps.

The key principle is that domain knowledge supplies the problems, while a
carefully designed verifier turns those problems into useful and trustworthy
GRPO learning signals.
