# 🛡️ Sentinel Sandbox

> **Status:** Active Research (v0.1)
> **Focus:** Linux Kernel, ptrace, Syscall Analysis, MLSecOps
> **Maintainer:** Nevin Shine 

## 📜 Overview
**Sentinel Sandbox** is a lightweight, custom-built runtime analysis environment designed to detect malicious behavior in Linux binaries. Unlike traditional sandboxes that rely on heavy virtualization (Cuckoo/VMware), Sentinel uses the Linux `ptrace` API to intercept system calls in real-time with minimal overhead.

**Research Goal:** To investigate the efficacy of **Dynamic Weightless Neural Networks (DWN)** for detecting zero-day anomalies in system call traces on resource-constrained (CPU-only) environments.

## 🏗️ Architecture
The system operates on three distinct layers:
1.  **The Interceptor (C/Kernel):** A custom tracer using `ptrace` to halt process execution at specific syscall entry/exit points.
2.  **The Cage (Docker):** An isolated container environment where the untrusted binary executes.
3.  **The Brain (Python/PyTorch):** A custom **CPU-Optimized Weightless Neural Network** (based on WiSARD) that detects anomalies without requiring GPUs.

## 🗺️ Roadmap
- [ ] **Phase 1: The Loader** (Process instantiation, memory mapping, & PID control)
- [ ] **Phase 2: The Tracer** (Intercepting `execve`, `open`, `write` syscalls)
- [ ] **Phase 3: The Logger** (Structured logging of register states & arguments)
- [x] **Phase 4: ML Integration** (Implemented Custom Weightless Neural Network on UNSW-NB15 Data) ✅ *Completed Day 8*

## 🛠️ Tech Stack
* **Language:** C (Core Logic), Python (Data Analysis/ML)
* **Kernel APIs:** `ptrace`, `waitpid`, `user_regs_struct`
* **ML Architecture:** Dynamic Weightless Networks (DWN), PyTorch Embeddings
* **OS:** Ubuntu 24.04 LTS (Hardened Kernel)

---
*Built by [Nevin Shine](https://www.linkedin.com/in/nevin-shine-b403b932b/) - Independent Researcher, Systems Security.*
