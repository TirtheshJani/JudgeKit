# Phase 5: Remaining Benchmarks

**Date:** 2026-05-17  
**Goal:** MedQA, MMLU clinical, HumanEval, MBPP adapters all implemented with TDD, matching the PubMedQA pattern. Runner already imports them; tests go in `tests/test_benchmarks/`.

## Architecture

Four independent adapters, all following the exact same pattern as `PubMedQAAdapter`:
- `load_dataset` call patched in tests (no real HF downloads in CI)
- `itertools.islice` for n-limiting
- `BenchmarkItem` with id, question, reference_answer, metadata

## Dataset field mapping

### MedQA (`GBaker/MedQA-USMLE-4-options`, split="test")
| HF field | BenchmarkItem field |
|----------|---------------------|
| index (enumerate) | id |
| `question` + formatted `options` | question |
| `answer_idx` ("A"/"B"/"C"/"D") | reference_answer |
| `{"options": options, "candidate_answer": answer_text}` | metadata |

### MMLU Clinical (`cais/mmlu`, 4 subjects, split="test")
| HF field | BenchmarkItem field |
|----------|---------------------|
| `f"{subject}_{index}"` | id |
| `question` + formatted `choices` | question |
| `["A","B","C","D"][answer]` | reference_answer |
| `{"choices": choices, "subject": subject}` | metadata |

### HumanEval (`openai_humaneval`, split="test")
| HF field | BenchmarkItem field |
|----------|---------------------|
| `task_id` | id |
| `prompt` | question |
| `canonical_solution` | reference_answer |
| `{"entry_point": ..., "test": ...}` | metadata |

### MBPP (`mbpp`, split="test")
| HF field | BenchmarkItem field |
|----------|---------------------|
| str(`task_id`) | id |
| `text` | question |
| `code` | reference_answer |
| `{"test_list": test_list}` | metadata |

## Tasks (all 4 parallel, non-overlapping files)

Each agent: write 6 tests → watch fail → implement → watch pass → commit → push.

### Agent MedQA
- `src/judgekit/benchmarks/medqa.py`
- `tests/test_benchmarks/test_medqa.py`

### Agent MMLU
- `src/judgekit/benchmarks/mmlu_clinical.py`
- `tests/test_benchmarks/test_mmlu_clinical.py`

### Agent HumanEval
- `src/judgekit/benchmarks/humaneval.py`
- `tests/test_benchmarks/test_humaneval.py`

### Agent MBPP
- `src/judgekit/benchmarks/mbpp.py`
- `tests/test_benchmarks/test_mbpp.py`

## Gate

- All 4 adapter test files have at least 6 passing tests each
- Full suite passes (no regressions)
- `uv run ruff check .` clean
