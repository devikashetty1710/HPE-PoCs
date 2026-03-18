"""
agent/agent_core.py

Builds and returns the LangChain AgentExecutor.

Architecture decisions:
  - ReAct pattern: Thought → Action → Observation → repeat → Final Answer
  - LOCAL-FIRST system prompt: instructs the LLM to always try
    LocalDocumentSearch before any web tool.
  - Supports Gemini, OpenAI, and Groq via LLM_PROVIDER config.
  - verbose=True so you see the full reasoning trace in the terminal.
"""

import logging
import os
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System / ReAct prompt
# ------------------------------------------------------------------

LOCAL_FIRST_REACT_PROMPT = """You are a document retrieval assistant.

STRICT RULES:
1. Always call LocalDocumentSearch FIRST.
2. After ANY observation — write Final Answer immediately. Never write the answer inside Thought.
3. If LocalDocumentSearch returns NOT_FOUND_LOCALLY — call WikipediaSearch once then Final Answer.
4. Never call the same tool twice.
5. Never put the answer text inside a Thought. Only put reasoning in Thought.

Available tools:
{tools}

FORMAT — follow this EXACTLY every single time, no exceptions:

Question: the input question
Thought: (only your reasoning here, never the answer)
Action: (tool name from [{tool_names}])
Action Input: (input to the tool)
Observation: (tool result — do not write this yourself)
Thought: I now have the answer.
Final Answer: (your answer here, mention the source file or Wikipedia)

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# ------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------

def _build_llm():
    """Instantiate the LLM based on LLM_PROVIDER setting."""
    if settings.LLM_PROVIDER == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info("Using Gemini model: %s", settings.GEMINI_MODEL)
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0,
                convert_system_message_to_human=True,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            )

    elif settings.LLM_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI

            logger.info("Using OpenAI model: %s", settings.OPENAI_MODEL)
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0,
            )
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            )

    elif settings.LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq

            groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            logger.info("Using Groq model: %s", groq_model)
            return ChatGroq(
                model=groq_model,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
            )
        except ImportError:
            raise ImportError(
                "langchain-groq is not installed. "
                "Run: pip install langchain-groq"
            )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
            "Set LLM_PROVIDER to 'gemini', 'openai', or 'groq' in your .env file."
        )


# ------------------------------------------------------------------
# Agent builder
# ------------------------------------------------------------------

def build_agent() -> AgentExecutor:
    """
    Build and return the LangChain AgentExecutor.

    Steps:
      1. Validate environment (API keys present).
      2. Instantiate the LLM.
      3. Build the ReAct prompt with local-first instructions.
      4. Create the ReAct agent.
      5. Wrap in AgentExecutor with error handling and iteration limits.
    """
    settings.validate()

    llm = _build_llm()
    tools = ALL_TOOLS

    prompt = PromptTemplate.from_template(LOCAL_FIRST_REACT_PROMPT)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.VERBOSE,
        max_iterations=15,                      # increased from 4
        max_execution_time=120,                 # 120 second hard timeout
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(
        "Agent ready. LLM=%s | Tools=%s | MaxIterations=%d",
        settings.LLM_PROVIDER,
        [t.name for t in tools],
        15,
    )
    return executor