# System Architecture Overview

This document provides a comprehensive visual and conceptual overview of the MMAI-AGI Framework architecture, addressing reviewer feedback for enhanced accessibility.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MMAI-AGI FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  STRATEGIC      │  │   INFORMATION   │  │    TEMPORAL     │  │
│  │    FIELDS       │◄─┤   RETRIEVAL     ├─►│  ARCHITECTURE   │  │
│  │                 │  │    NETWORK      │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │         │
│           ▼                     ▼                     ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     AGENTS      │  │     MEMORY      │  │      NASH       │  │
│  │  (Cognitive     │◄─┤    SYSTEMS      ├─►│   EQUILIBRIUM   │  │
│  │   Elements)     │  │                 │  │   VALIDATOR     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │         │
│           └─────────────────────┼─────────────────────┘         │
│                                 ▼                               │
│                    ┌─────────────────┐                         │
│                    │   ENVIRONMENT   │                         │
│                    │     SYSTEM      │                         │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Strategic Fields
**Purpose**: Wave-like propagation of strategic information through space

```
Strategic Field Structure:
┌─────────────────────────────────────┐
│  φᵢ(x,t) = Aᵢ φᵢ(x) exp(-iEᵢt/ℏ)  │
├─────────────────────────────────────┤
│  Components:                        │
│  • Amplitude (Aᵢ): Strategy strength│
│  • Position (x): Spatial location   │
│  • Time (t): Temporal evolution     │
│  • Energy (Eᵢ): Strategic potential │
└─────────────────────────────────────┘
```

**Implementation**: `simulation/core/strategic_field.py`

### 2. Information Retrieval Network (IRN)
**Purpose**: Dual individual/collective memory system enabling stigmergic coordination

```
IRN Architecture:
┌─────────────────────────────────────┐
│           COLLECTIVE MEMORY         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │Frame│ │Frame│ │Frame│ │Frame│   │
│  │  1  │ │  2  │ │  3  │ │  N  │   │
│  └─────┘ └─────┘ └─────┘ └─────┘   │
├─────────────────────────────────────┤
│          INDIVIDUAL MEMORIES        │
│ Agent1   Agent2   Agent3   AgentN   │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│ │Local│ │Local│ │Local│ │Local│    │
│ │Frame│ │Frame│ │Frame│ │Frame│    │
│ └─────┘ └─────┘ └─────┘ └─────┘    │
└─────────────────────────────────────┘
```

**Implementation**: `simulation/core/memory_system.py`

### 3. Fractal Time Architecture
**Purpose**: Multi-scale temporal coordination across dt, t, and T scales

```
Temporal Hierarchy:
T-Scale (Global Behavior)
├── t-Scale (Pattern Formation)
│   ├── dt-Scale (Individual Decisions)
│   ├── dt-Scale (Individual Decisions)
│   └── dt-Scale (Individual Decisions)
├── t-Scale (Pattern Formation)
│   ├── dt-Scale (Individual Decisions)
│   ├── dt-Scale (Individual Decisions)
│   └── dt-Scale (Individual Decisions)
└── t-Scale (Pattern Formation)
    ├── dt-Scale (Individual Decisions)
    ├── dt-Scale (Individual Decisions)
    └── dt-Scale (Individual Decisions)
```

**Implementation**: `simulation/core/fractal_time_manager.py`

### 4. Cognitive Elements (Agents)
**Purpose**: Individual decision-making units with strategic capabilities

```
Agent Architecture:
┌─────────────────────────────────────┐
│         CEREBRAL UNIT OF            │
│         INTELLIGENCE (CUI)          │
├─────────────────────────────────────┤
│  ┌─────────────────────────────────┐│
│  │      PERCEPTION STAGE           ││
│  │  • Environmental sensing        ││
│  │  • Strategic field detection    ││
│  │  • Memory activation            ││
│  └─────────────────────────────────┘│
│  ┌─────────────────────────────────┐│
│  │       ANALYSIS STAGE            ││
│  │  • Similarity engine (S)        ││
│  │  • Difference engine (D)        ││
│  │  • Frame system processing      ││
│  └─────────────────────────────────┘│
│  ┌─────────────────────────────────┐│
│  │        ACTION STAGE             ││
│  │  • Strategy selection           ││
│  │  • Field influence              ││
│  │  • Memory updating              ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

**Implementation**: `simulation/core/agent.py`

## Information Flow

### 1. Perception-Analysis-Action (PAA) Cycle

```
Environmental Input
        │
        ▼
┌─────────────────┐
│   PERCEPTION    │ ◄─── Strategic Field
│                 │ ◄─── Memory (IRN)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    ANALYSIS     │ ◄─── Similarity Engine
│                 │ ◄─── Difference Engine
└─────────────────┘
        │
        ▼
┌─────────────────┐
│     ACTION      │ ────► Strategic Choice
│                 │ ────► Field Influence
└─────────────────┘ ────► Memory Update
```

### 2. Strategic Field Propagation

```
Agent Decision
     │
     ▼
Local Field Influence
     │
     ▼
Wave Propagation (Diffusion)
     │
     ▼
Spatial Pattern Formation
     │
     ▼
Global Field Coherence
     │
     ▼
Collective Behavior Emergence
```

### 3. Memory Integration

```
Individual Experience
        │
        ▼
Local Frame Creation
        │
        ▼
Importance Evaluation
        │
        ▼ (if important)
Collective Memory Integration
        │
        ▼
Stigmergic Coordination
        │
        ▼
Enhanced Decision Making
```

## Mathematical Relationships

### Core Equations

1. **Strategic Field Evolution**:
   ```
   ∂φᵢ/∂t = D∇²φᵢ + Σⱼ Iⱼ(x,t)
   ```

2. **Nash Equilibrium Convergence**:
   ```
   S∞ = lim[t→∞] ∫ PPP(s,p,𝒢) dt
   ```

3. **Memory Integration**:
   ```
   F_collective = ∫ F_individual dt
   ```

4. **Temporal Resonance**:
   ```
   T = ∫ t = ∫∫ dt
   ```

## Implementation Mapping

| Component | File Location | Key Classes |
|-----------|---------------|-------------|
| Strategic Fields | `core/strategic_field.py` | `StrategicField` |
| IRN | `core/memory_system.py` | `MemorySystem` |
| Fractal Time | `core/fractal_time_manager.py` | `FractalTimeManager` |
| Agents | `core/agent.py` | `Agent` |
| Nash Validation | `core/nash_validator.py` | `NashValidator` |
| Environment | `core/environment_system.py` | `EnvironmentSystem` |

## Experimental Validation

The architecture is validated through five key experiments:

1. **Nash Equilibrium**: Validates convergence properties
2. **Strategic Fields**: Confirms wave-like propagation
3. **Fractal Time**: Tests multi-scale coordination
4. **Hypersensitive Points**: Examines decision sensitivity
5. **Stigmergic Coordination**: Validates memory integration

## Comparison with Existing AGI Approaches

| Approach | Architecture | Key Difference |
|----------|-------------|----------------|
| **Neural Networks** | Centralized processing | MMAI: Distributed strategic fields |
| **Symbolic AI** | Rule-based reasoning | MMAI: Emergent strategic behavior |
| **Reinforcement Learning** | Individual optimization | MMAI: Collective equilibrium |
| **Large Language Models** | Transformer architecture | MMAI: Wave-based information flow |
| **Embodied Cognition** | Sensorimotor grounding | MMAI: Strategic field grounding |

## Scalability Considerations

The architecture scales through:

- **Spatial Scaling**: Strategic fields expand to larger grids
- **Population Scaling**: More agents increase field complexity
- **Temporal Scaling**: Deeper fractal time hierarchies
- **Strategic Scaling**: Higher-dimensional strategy spaces

For implementation details, see the complete simulation framework in this repository.
