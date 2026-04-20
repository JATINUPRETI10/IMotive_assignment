# AI Cognitive Routing & RAG Assignment

## Overview
This project implements a basic system where posts are routed to different personas and responses are generated using a step-by-step pipeline.

The assignment is divided into three parts:
- Routing posts to relevant bots
- Generating content using a structured flow
- Handling replies with full context and resisting prompt injection

---

## Phase 1: Persona Routing

- Used FAISS as an in-memory vector store
- Converted persona descriptions into embeddings using sentence-transformers
- Compared incoming post with personas using cosine similarity
- Selected the most relevant bot based on similarity score

---

## Phase 2: LangGraph Flow

The content generation follows three steps:

1. Decide Search  
   The system converts the topic into a short search query  

2. Mock Search  
   A simple function returns relevant news based on keywords  

3. Generate Post  
   The LLM generates a post based on persona and context  

---

## Phase 3: RAG + Defense

- The full conversation (parent post + history + reply) is passed to the model
- The system prompt ensures:
  - The bot stays in its persona
  - Malicious instructions are ignored
- The model continues the argument instead of following injected instructions

---

## Tech Stack

- Python  
- FAISS  
- Sentence Transformers  
- LangGraph  
- Groq API  

---

## Project Structure

IMotive_assignment/
- router.py
- main.py
- rag.py
- logs.md
- requirements.txt
- .env.example

---

## Notes

- API keys are handled using environment variables
- The implementation focuses on clarity and correctness
- All required phases from the assignment are covered
