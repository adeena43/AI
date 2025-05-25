### Q1: Brief Responses

---

### 1. **What are the primary characteristics that define a rational agent in Artificial Intelligence?**

* A **rational agent** selects actions that maximize its **expected performance measure** based on the **percept history** and **built-in knowledge**.
* Key characteristics:

  * **Autonomy**
  * **Perception**
  * **Goal-oriented behavior**
  * **Ability to learn and adapt**

---

### 2. **Why do CSPs use factored state representation? Justify.**

* **Factored state representation** breaks down the overall state into variables and their domains.
* Justification:

  * Enables **efficient constraint checking**.
  * Simplifies representation and supports **constraint propagation**.
  * Helps in **independently manipulating** parts of the problem.

---

### 3. **Is the drone delivery environment episodic or sequential?**

* **Sequential.**
* Justification: Each decision (e.g., path taken, package dropped) **affects future states**, delivery outcomes, and subsequent actions. The agent must **remember past decisions** to make informed future choices.

---

### 4. **How does local beam search differ from standard hill-climbing?**

* **Local beam search** maintains **k states simultaneously**, whereas **hill-climbing** operates on **a single state**.
* Beam search uses **collective progress** to avoid local maxima by tracking **multiple paths**, while hill-climbing can get stuck on **plateaus or ridges**.

---

### 5. **DFS vs BFS – Space and Time Complexity**

| Algorithm | Time Complexity  | Space Complexity |
| --------- | ---------------- | ---------------- |
| **DFS**   | O(b<sup>m</sup>) | O(bm)            |
| **BFS**   | O(b<sup>d</sup>) | O(b<sup>d</sup>) |

* **b:** branching factor, **d:** depth of shallowest goal, **m:** max depth of tree.

---

### 6. **Manhattan distance as admissible heuristic for rook moves – (T/F)**

* **False.**
* Justification: Rooks move in **straight lines**, and each move can traverse **multiple squares**. **Manhattan distance counts steps**, not moves. Therefore, it **overestimates** in some cases — violating admissibility.

---

### 7. **Is minimax only optimal when the opponent is also optimal?**

* **True.**
* Justification: Minimax assumes the opponent will make **optimal (worst-case)** moves. If the opponent is irrational or makes random choices, **other strategies** (like expected utility or probabilistic reasoning) may yield better outcomes.

---

### 8. **Two Limitations of Hill Climbing Algorithm**

1. **Gets stuck in local maxima** — it can’t explore beyond nearby states.
2. **Plateaus and ridges** cause the algorithm to make **no progress** or **oscillate** without improvement.

---

### 9. **Size of CSP search tree**

* (i) **With commutativity:**

  $$
  \text{Size} = d^n / n!
  $$

  (order doesn't matter for assignment)

* (ii) **Without commutativity:**

  $$
  \text{Size} = d^n
  $$

  (order of variable assignments matters)

---

### 10. **Can iterative deepening be applied to adversarial search?**

* **Yes.**
* Justification: **Iterative Deepening Minimax** (used in games like chess) combines depth-first traversal with depth limits, allowing:

  * **Any-time behavior**
  * **Optimal move discovery within time constraints**
  * Balanced **memory usage** and **decision quality**


