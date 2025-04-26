# Model Evaluation and Smart Analytics (MESA) Tool

## 1. Motivation

Large Language Models (LLMs) demand vast computing resources and incur significant costs, even just for experimental runs. To optimize both performance and total cost of ownership (TCO), architecture and engineering teams must:
- Gain comprehensive visibility across every stage of the inference and training pipeline  
- Identify computational and memory hot-spots  
- Rapidly assess trade-offs between throughput, cost, accuracy, and more  

This tool will enable interactive “what-if” exploration of deployment strategies—such as parallelism and concurrency—revealing where and how optimizations can deliver the greatest ROI.

---

## 2. Function Requirements

### 2.1 Flexible Model Description
- **Evolving Model Architectures**  
  Design a schema that can express transformer-style blocks, multi-head attention modules, MLP layers, embeddings, adapters, etc., and easily extend to accommodate future innovations.  
- **Parameter Metadata**  
  Capture comprehensive model metadata, including tensor shapes, data types, density and sparsity profiles, and quantization scales to enable precise cost and complexity analysis.

### 2.2 Hardware Specification
- **Abstract Hardware Profiles**  
  Define a plug-and-play descriptor for compute devices that captures peak FLOPS, memory capacity, and interconnect topology.  
- **Hierarchical Resource Model**  
  Represent multi-level memory and compute resources as a graph
  
### 2.3 Detailed Interaction Diagrams
- **Component-Level Data Flows**  
  Auto-generate diagrams showing how inputs, activations, weights and output move through the transformer stack and pipeline.  
- **Dependency Graphs**  
  Highlight control-flow and data-dependency chains to analyze serialization points and parallelization barriers.

### 2.4 Fine-Grained Cost Metrics
- **Computational Complexity**  
  Report FLOPs per layer and aggregate FLOPs for end-to-end execution.  
- **Memory Footprint**  
  Measure peak and working-set usage across model state, activations, optimizer state (for training), and intermediate buffers.  
- **Latency Breakdown**  
  Provide per-layer and per-stage latencies, including data transfers, compute, and I/O.

### 2.5 Interactive Optimization Playground
- **Parameter Sweeps**  
  Adjust batch size, sequence length, precision, and degrees of parallelism on the fly.  
- **Impact Visualization**  
  Deliver real-time feedback on throughput, latency, resource utilization, and cost estimates.

---

## 3. Installation and Setup

To get the tool up and running, follow these steps in order.  
We assume you have Git, and Node.js (14+) installed.  
If not, install them first.

### 3.1 Prerequisites

#### Clone the repository

```bash
git clone https://github.com/chamidou2k/mesa.git
cd mesa
```


### 3.2 Back-End Setup


#### Install Dependencies
Navigate to the backend directory and install required packages via npm:
```bash
cd ../backend
npm install 
```
#### Configure Back-end API Service Port
The default service port for the back-end API is 9090.
You can also specify a custom port using the PORT environment variable:
```bash
PORT=9090  npm run dev
```

#### Start the Back-End Service

```bash
npm run dev
```
#### Service Endpoint Verification
Successful initialization will display:
```plaintext
> js_backend@1.0.0 dev
> node index.js

MESA Tool Backend running at localhost:9090
```

### 3.3 Front-End Setup

#### Install dependencies
Navigate to the frontend directory and install required packages via npm:
```bash
cd ../frontend
npm install
```
#### Configure Backend API address
Update backend API address in `.env` file:
```env
VITE_BACKEND_API_BASE=http://localhost:9090
```
#### Configure Service Port
Modify vite.config.js to customize the development server settings:

```javascript
server: {
	port: 9000,
	host: false,
},    
```
#### Launch Development Server

```bash
npm run dev
```
#### Service Endpoint Verification
Successful initialization will display:
```plaintext
VITE v5.4.10  ready in 270 ms
  ➜  Local:   http://localhost:9000/
  ➜  press h + enter to show help
```
---
## 4. Latest Status

### Model Architecture Specification
- Defined a unified metadata schema for describing model architectures.

### Model Parameter Binding Mechanism
- Implemented an automated parameter-binding system to synchronize metadata descriptions with Hugging Face model parameter files.

### Basic Hardware Specification Support
- Implemented general hardware specification support.

### LLaMA2 (LlamaForCausalLM) Verification
- Completed operator testing for the LlamaForCausalLM architecture.

### Basic UI Framework
- Developed core UI components:
  - Configuration panel
  - Interactive graph canvas
  - Runtime metrics display panel

---

## 5. Next Steps (rev 1.#)

### Expand Model Support
- Add general Mixture-of-Experts (MoE) model compatibility.
- Integrate Deepseek v3 and LLaMA v4 model architecture.

### Enhance UI Experience
- Refine interface workflows and interactions.
- Improve visual consistency and responsiveness.

### Add Training Pipeline Support
- Expand to cover training pipeline.

---

## Note for Tool Usage 
The primary goal of this tool is to explore a general framework for evaluating the LLM pipeline, focusing on latency, memory footprint, and FLOPs analysis,
while maintaining scalability to support the continuous evolution of model architectures.
There are known limitations in the current evaluation and calculation methods, due to both the limited availability of test data and the still-maturing understanding of model knowledge. As the tool continues to develop, the accuracy and reliability of the results are expected to improve. For now, please take this tool for experimental purposes only, and interpret the evaluation results with caution.

"The more I learn, the more I realize how much I don’t know." — This quote perfectly captures the sentiment around the rapid and evolving landscape of AI.

---

