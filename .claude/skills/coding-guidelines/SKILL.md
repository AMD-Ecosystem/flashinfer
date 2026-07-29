---
name: coding-guidelines
description: Disciplined coding guidelines for amd-flashinfer changes — think before coding, prefer simplicity, make surgical changes, and drive execution against verifiable goals. Load when starting an implementation task or when the user asks for the coding guidelines.
---

# Coding Guidelines

## 1 — Think Before Coding

Before generating code, carefully reason about the problem.
State assumptions explicitly. If the requirements, scope, or desired outcome are unclear, ask for clarification. Do not proceed with guesswork.
Push back when a simpler approach exists.
Stop when confused.

## 2 — Simplicity First

Avoid unnecessary over-engineering. Write clean, readable, and simple code.
Choose the most straightforward implementation over complex optimizations, unless performance is strictly required.
Do not use abstractions for single-use code.

## 3 — Surgical Changes

Modify only what is necessary to fulfill the goal. Do not touch or refactor surrounding code unless it directly solves the task at hand.
Match existing style.
Do not refactor what isn't broken.

## 4 — Goal-Driven Execution

Break tasks down into verifiable steps.
Before writing code, outline the plan and define success criteria. Verify your edits against these defined goals.
Loop until verified.
