# Environment Types in AI

This document explains the different types of environments an AI agent can operate in. Each type is described in simple terms with real-life examples.

---

## 1) Fully Observable vs. Partially Observable

- **Fully Observable**: The agent has complete information about the environment.
  - ✅ *Example*: Playing **chess** – you can see all pieces on the board.
- **Partially Observable**: The agent has limited or incomplete information.
  - ❌ *Example*: **Driving in fog** – you can’t see everything clearly.

---

## 2) Deterministic vs. Stochastic (Non-deterministic)

- **Deterministic**: The result of every action is predictable.
  - ✅ *Example*: A **calculator** – `2 + 2` always gives `4`.
- **Stochastic**: Actions have unpredictable outcomes due to randomness.
  - ❌ *Example*: **Rolling a dice** – the outcome is uncertain.

---

## 3) Episodic vs. Sequential

- **Episodic**: Each action is independent; past actions don't affect the next.
  - ✅ *Example*: **Image recognition** – each image is analyzed independently.
- **Sequential**: Past actions affect future ones.
  - ❌ *Example*: **Cooking** – what you do early affects later steps.

---

## 4) Static vs. Dynamic

- **Static**: The environment does not change while the agent is making a decision.
  - ✅ *Example*: **Sudoku puzzle** – it remains the same as you solve it.
- **Dynamic**: The environment may change during the agent's reasoning.
  - ❌ *Example*: **Stock market** – prices change in real time.

---

## 5) Discrete vs. Continuous

- **Discrete**: There are a limited number of distinct actions or inputs.
  - ✅ *Example*: **Board games** – a finite set of moves.
- **Continuous**: Infinite number of possibilities for actions or inputs.
  - ❌ *Example*: **Driving a car** – steering and speed can vary continuously.

---

## 6) Single Agent vs. Multiple Agents

- **Single Agent**: One agent interacts with the environment.
  - ✅ *Example*: **Solving a puzzle** by yourself.
- **Multiple Agents**: Many agents interacting, either competitively or cooperatively.
  - ❌ *Example*: **Football match** – many players involved.

---

## Summary Table

| Type                        | Explanation                            | Example                     |
|-----------------------------|----------------------------------------|-----------------------------|
| Fully vs. Partially Observable | Full vs. Partial view                 | Chess vs. Driving in fog    |
| Deterministic vs. Stochastic  | Predictable vs. Random results        | Calculator vs. Dice roll    |
| Episodic vs. Sequential       | Independent vs. Dependent steps       | Image tagging vs. Cooking   |
| Static vs. Dynamic            | No change vs. Changing environment    | Sudoku vs. Stock market     |
| Discrete vs. Continuous       | Limited vs. Infinite actions          | Chess vs. Car driving       |
| Single vs. Multiple Agents    | One vs. Many interacting agents       | Puzzle vs. Multiplayer game |

---

> 📘 **Note:** Understanding these types helps design better AI agents based on the nature of the task and environment.

