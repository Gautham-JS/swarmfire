# Swarmfire: Gated Transformer XL Architecture

Welcome to the repo for Swarmfire. This project implements a specialized reinforcement learning agent designed for partially observable environments using a highly modified GTrXL (Gated Transformer XL) architecture called GTrXLH, Gated Transformer XL Hyperconnected.

## The Model: GTRXLH: GTrXL with Hyperconnections

The core of our agent is built on a custom transformer backbone that focuses on maintaining long-term dependencies without the usual representational collapse seen in deep Transformers. I've implemented several key architectural features to ensure the model can effectively utilize its memory over long rollout horizons.

This model is defined in `policies/TrXL.py`. This name needs update though as it still sounds like a normal TransformerXL model.


### Key Architectural Features

* **Gated Residual Connections**: Instead of standard residual connections, I use a learned gating mechanism. This allows the model to dynamically decide how much new information from the sublayer should be integrated into the existing hidden state, helping to stabilize training in partially observable settings.
* **Hyperconnections (Cross-Layer Mixing)**: To prevent the network from becoming too deep and losing signal, I've implemented hyperconnections. This feature allows each layer to perform a learned, softmax-weighted sum of all prior hidden states plus its own sublayer output. It basically creates dynamic, dense residual connections across the entire depth of the network.
* **Depth Blending**: By combining gating and hyperconnections, the model can effectively "blend" features across different depths, making it much more robust to the vanishing gradient problems as well as representational collapse that often plague deep recurrent/transformer models in unsupervised tasks.
* **Relative Positional Encoding**: I use a fixed, sinusoidal relative positional encoding. Because it's based on relative distance rather than absolute time, the model doesn't experience any drift as tokens age through the sliding window memory buffer.

<!-- ARCHITECTURAL DIAGRAM PLACEHOLDER -->
<!-- [Insert Architecture Diagram Here] -->

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.]('policies/HC_Updated.drawio (2).png')

## Practical Application: Sim-to-World Transfer

One of our primary goals is making it as easy as possible to move an agent from simulation to a real-world or high-fidelity environment. I've built the transport and interface layers to be extremely flexible.

### Decoupled Interface Design

The agent doesn't care where its observations come from or where its actions go, as long as they follow the required interface. This abstraction allows for clean separation of concerns:

* **Communication Layers**: Currently, I use a WebSocket server (`comms/web_sockets/server.py`) to facilitate communication between different parts of the system.
* **Data Pipeline**: Our environment implementation (`envs/RedisSingleAgentEnv.py`) uses a Redis client to publish environmental data to a message queue. This allows multiple observers or rendering processes to subscribe to the same stream without slowing down the training loop.
* **Extensibility**: The interface is designed so that an agent can simply implement the required methods to interface with new systems, such as ROS2, for robot deployment.

### Unreal Engine 5 Integration

I am currently demonstrating this setup by running an agent abstraction inside a high-fidelity Unreal Engine 5 simulation. In this workflow:
1. The **UE5 Simulation** computes the physics and visual data.
2. Data is published via **WebSockets** and/or **Redis**.
3. The **RL Agent** consumes these observations, computes actions, and sends them back through the same pipeline.

This setup provides a seamless bridge between the high-level reasoning of the GTrXL model and the complex, low-level physics of advanced simulators, paving the way for robust sim-to-world transfer.
