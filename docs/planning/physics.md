# Supply Chain Physics: Theory, Application, and Validation Rubric

Supply Chain Physics (SCP) is the application of manufacturing science and queuing theory to the movement of goods, information, and capital. Unlike traditional "optimization," SCP focuses on the governing laws that determine what is physically possible within a network.

## 1. Terms of Art

*   **WIP (Work-in-Process):** The total inventory currently within the boundaries of a process or network.
*   **Throughput (TH):** The average output rate of a process (units/time).
*   **Cycle Time (CT):** The total time a unit spends in the system from start to finish.
*   **Utilization ($u$):** The ratio of the arrival rate to the maximum capacity ($u = \lambda / \mu$).
*   **Variability ($V$):** Random fluctuations in arrival times ($c_a$) or process times ($c_e$).
*   **Mass Balance:** The conservation principle stating that matter is neither created nor destroyed; inventory at time $t$ must equal $I_{t-1} + \text{Inflow} - \text{Outflow}$.
*   **Signal Resonance (Bullwhip Effect):** The phenomenon where small fluctuations in consumer demand result in increasingly larger fluctuations in demand at the supplier levels.

## 2. Core Theoretical Frameworks

### I. Little’s Law (The Fundamental Constraint)
The most basic "law of gravity" in supply chains. It states that for any stable system:

$$ 
WIP = TH \times CT 
$$

**LLM Validation Rule:** Synthetic data must maintain this relationship. If a generated dataset reports a $TH$ of 10 units/day and a $CT$ of 5 days, the $WIP$ must be 50. Any deviation indicates "physical leaks" or "teleportation."

### II. The VUT Equation (Kingman’s Formula)
Describes why lead times explode as systems approach full capacity.

$$ 
CT_q \approx \left( \frac{c_a^2 + c_e^2}{2} \right) \left( \frac{u}{1-u} \right) t_e 
$$

*   **V (Variability):** $\frac{c_a^2 + c_e^2}{2}$
*   **U (Utilization):** $\frac{u}{1-u}$
*   **T (Time):** $t_e$ (Mean processing time)

**LLM Validation Rule:** Cycle time is non-linear. As utilization ($u$) approaches 1.0 (100%), Cycle Time ($CT$) must increase exponentially.

### III. Conservation of Flow (Mass Balance)
Inventory levels must be mathematically reconciled across time intervals.

$$ 
I_{t} = I_{t-1} + \text{Receipts}_{t} - \text{Shipments}_{t} 
$$

**LLM Validation Rule:** Every change in inventory state must be tied to a transaction (PO Receipt, Sales Order, or Scrap). Inventory cannot "evaporate" or "materialize" without a balancing friction record.

### IV. The Law of Variability & Signal Resonance
Variability always degrades performance and amplifies as it moves upstream.

*   **Buffering:** A system buffers variability through Inventory, Capacity, or Time.
*   **Bullwhip Logic:** The Coefficient of Variation ($CV$) of orders at the supplier level must be greater than or equal to the $CV$ of demand at the retail level.

## 3. Application: The Internal Benchmarks

*   **The Best Case (Penny Fab One):** Zero variability. $WIP$ follows a linear path until it hits "Critical WIP" ($W_0$).
*   **The Practical Worst Case (PWC):** Represents a system with high (Poisson) randomness. $CT$ increases significantly before reaching full capacity.
*   **The Absolute Worst Case:** Units move in a single-file line with maximum possible delay.

## 4. Validation Rubric for Synthetic Data

| Element | Requirement | Violation Signal (Red Flag) |
| :--- | :--- | :--- |
| **Kinematic Consistency** | $WIP = TH \times CT$ | $WIP$ is disconnected from the product of rate and time. |
| **Mass Balance** | $I_t = I_{t-1} + R_t - S_t$ | Inventory "teleports" or "evaporates" without transaction logs. |
| **Thermal Congestion** | Non-linear $CT$ at $u > 85\%$ | Lead times remain linear or static during high-utilization periods. |
| **Signal Resonance** | $CV_{\text{upstream}} \geq CV_{\text{downstream}}$ | Demand signals are equally "smooth" across the entire chain. |
| **Lead Time Tail** | Cycle times show a "fat tail" | Normal/Gaussian distribution of lead times in a high-utilization system. |
| **Variance Coupling** | Downstream $c_a \geq$ Upstream $c_e$ | Downstream arrival variability is lower than the source process variability. |

## 5. LLM Implementation Prompting (Constraint Snippet)

> "Ensure all generated metrics adhere to Supply Chain Physics:
> *   Maintain Little’s Law ($WIP = TH \times CT$) at all nodes.
> *   Implement Mass Balance: $I_t = I_{t-1} + \text{Inflow} - \text{Outflow}$.
> *   Apply VUT Logic: As utilization exceeds 85%, cycle time must increase non-linearly.
> *   Model Signal Resonance: Ensure the Coefficient of Variation ($CV$) of orders increases as data moves from Retail to Tier 2 Suppliers."
