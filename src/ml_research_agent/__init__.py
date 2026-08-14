"""ml-research-agent: an agentic system for end-to-end ML research.

Given a research idea in natural language, this package runs a pipeline of
specialized agents that (1) survey the literature, (2) build a durable
knowledge base of papers + codebases, and (3) design, run, and analyze
experiments that test the idea.

Public surface (planned):
    from ml_research_agent import ResearchAgent, Config
    agent = ResearchAgent(Config.load())
    agent.run("Does curriculum ordering help small-model math reasoning?")
"""
