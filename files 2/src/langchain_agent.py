"""
LangChain agent module - enables natural language Q&A over the water quality dataframe.
Supports Groq (free), OpenAI, OpenRouter, and fallback rule-based mode.
"""

import os
import pandas as pd
from typing import Optional

try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from openrouter import OpenRouter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


WATER_QUALITY_CONTEXT = """
You are a water quality expert assistant analyzing a dataset with these columns:
- ph: pH level (safe range: 6.5-8.5)
- Hardness: Water hardness in mg/L
- Solids: Total dissolved solids in ppm (WHO limit: 500 ppm)
- Chloramines: Disinfectant level in ppm (safe: up to 4 ppm)
- Sulfate: Sulfate in mg/L (safe: up to 250 mg/L)
- Conductivity: Electrical conductivity in μS/cm (WHO limit: 400 μS/cm)
- Organic_carbon: Organic carbon in ppm (safe: < 2 ppm for treated water)
- Trihalomethanes: THMs in μg/L (safe: up to 80 μg/L)
- Turbidity: Turbidity in NTU (WHO limit: 5 NTU)
- Potability: Target variable (1 = safe to drink, 0 = not safe)

Provide clear, accurate answers based on the data. When relevant, reference WHO/EPA standards.
"""


class WaterQualityAgent:
    """LangChain-powered agent for water quality Q&A."""

    def __init__(self, df: pd.DataFrame, llm_provider: str = "groq",
                 api_key: Optional[str] = None, model_name: Optional[str] = None,
                 openrouter_model: Optional[str] = None):
        self.df = df
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.openrouter_model = openrouter_model
        self.agent = None
        self.llm = None
        self._initialize()

    def _initialize(self):
        """Initialize the LLM and pandas dataframe agent."""
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available. Using fallback mode.")
            return

        try:
            if self.llm_provider == "groq":
                api_key = self.api_key or os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not provided")
                self.llm = ChatGroq(
                    api_key=api_key,
                    model_name=self.model_name or "llama-3.3-70b-versatile",
                    temperature=0
                )

            elif self.llm_provider == "openai":
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not provided")
                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model=self.model_name or "gpt-4o-mini",
                    temperature=0
                )

            elif self.llm_provider == "openrouter":
                api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY not provided")
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model=self.openrouter_model or "meta-llama/llama-3.3-8b-instruct:free",
                    temperature=0,
                    default_headers={
                        "HTTP-Referer": "https://water-quality-ai.app",
                        "X-Title": "Water Quality Detection AI"
                    }
                )

            else:
                raise ValueError(f"Unsupported provider: {self.llm_provider}")

            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=False,
                agent_type="tool-calling",
                allow_dangerous_code=True,
                prefix=WATER_QUALITY_CONTEXT,
                max_iterations=8,
                handle_parsing_errors=True,
            )
        except Exception as e:
            print(f"Agent initialization failed: {e}")
            self.agent = None

    def ask(self, question: str) -> str:
        """Ask a natural language question about the data."""
        if self.agent is None:
            return self._fallback_answer(question)

        try:
            response = self.agent.invoke({"input": question})
            return response.get("output", str(response))
        except Exception as e:
            return f"Agent error: {e}\n\nFalling back to rule-based answer:\n{self._fallback_answer(question)}"

    def _fallback_answer(self, question: str) -> str:
        """Rule-based fallback when LLM is unavailable."""
        q = question.lower()
        df = self.df

        try:
            if "average" in q or "mean" in q:
                for col in df.select_dtypes(include="number").columns:
                    if col.lower() in q:
                        return f"The average {col} is {df[col].mean():.2f}"
                return f"Averages:\n{df.describe().loc['mean'].to_string()}"

            if "maximum" in q or "max" in q or "highest" in q:
                for col in df.select_dtypes(include="number").columns:
                    if col.lower() in q:
                        return f"The maximum {col} is {df[col].max():.2f}"

            if "minimum" in q or "min" in q or "lowest" in q:
                for col in df.select_dtypes(include="number").columns:
                    if col.lower() in q:
                        return f"The minimum {col} is {df[col].min():.2f}"

            if "how many" in q or "count" in q:
                if "safe" in q and "Potability" in df.columns:
                    return f"Safe samples: {(df['Potability'] == 1).sum()}"
                if "unsafe" in q and "Potability" in df.columns:
                    return f"Unsafe samples: {(df['Potability'] == 0).sum()}"
                return f"Total samples: {len(df)}"

            if "missing" in q or "null" in q:
                return f"Missing values per column:\n{df.isnull().sum().to_string()}"

            if "columns" in q or "features" in q:
                return f"Columns: {', '.join(df.columns.tolist())}"

            return (f"I can answer questions about averages, min/max, counts, and missing values. "
                    f"Dataset has {len(df)} rows and {len(df.columns)} columns: "
                    f"{', '.join(df.columns.tolist())}")
        except Exception as e:
            return f"Could not parse question. Error: {e}"


def create_agent(df: pd.DataFrame, provider: str = "groq",
                 api_key: Optional[str] = None,
                 openrouter_model: Optional[str] = None) -> WaterQualityAgent:
    """Factory function to create a water quality agent."""
    return WaterQualityAgent(df, llm_provider=provider, api_key=api_key,
                             openrouter_model=openrouter_model)