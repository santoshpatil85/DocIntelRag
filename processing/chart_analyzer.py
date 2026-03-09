"""
Chart and image analysis using GPT-4o Vision.

Analyzes charts, graphs, and visual elements to extract insights.
"""

import base64
import asyncio
from typing import Optional, Dict, Any
import io

from openai import AsyncAzureOpenAI
import aiohttp

from config.settings import settings
from utils.logger import setup_logger
from models.document import ChartAnalysis

logger = setup_logger(__name__)


class ChartAnalyzer:
    """
    Analyzes charts and visual elements using GPT-4o with vision.

    Extracts chart titles, trends, axes, legends, and key insights.
    """

    def __init__(self):
        """Initialize the Azure OpenAI client for vision analysis."""
        try:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure.openai_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=settings.azure.openai_endpoint,
            )
            logger.info("Initialized Azure OpenAI client for vision analysis")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    async def analyze_chart(
        self,
        image_bytes: bytes,
        context: Optional[str] = None,
    ) -> Optional[ChartAnalysis]:
        """
        Analyze a chart image using GPT-4o Vision.

        Args:
            image_bytes: Image data as bytes (PNG/JPEG).
            context: Optional context about the document or page.

        Returns:
            ChartAnalysis object or None if analysis fails.
        """
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Determine image media type (default to PNG)
            media_type = "image/png"

            # Prepare prompt
            system_prompt = """You are an expert data analyst specializing in chart 
and graph interpretation. Analyze the provided chart and extract:
1. Chart title
2. Chart type and description
3. Axis labels and their meanings
4. Key trends and patterns
5. Legend information if present
6. Key data insights

Format your response as structured data."""

            user_prompt = f"""Please analyze this chart and provide a detailed analysis.
Focus on:
- What is the main message of this chart?
- What are the key trends?
- What values are important?
- Are there any anomalies or noteworthy patterns?

{f"Additional context: {context}" if context else ""}

Provide the analysis in a clear, structured format."""

            # Call GPT-4o Vision
            response = await self.client.chat.completions.create(
                model=settings.azure.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    },
                ],
                max_tokens=1000,
            )

            # Parse response
            analysis_text = response.choices[0].message.content
            chart_analysis = self._parse_chart_analysis(analysis_text)

            logger.debug(f"Chart analysis completed: {chart_analysis.title}")
            return chart_analysis

        except Exception as e:
            logger.error(f"Failed to analyze chart: {str(e)}")
            return None

    def _parse_chart_analysis(self, analysis_text: str) -> ChartAnalysis:
        """
        Parse GPT response into structured ChartAnalysis.

        Args:
            analysis_text: Raw text response from GPT-4o.

        Returns:
            Structured ChartAnalysis object.
        """
        try:
            # Extract title (first line often contains it)
            lines = analysis_text.strip().split("\n")
            title = lines[0] if lines else "Untitled Chart"

            # Simple parsing - in production, use structured output or JSON mode
            description = analysis_text
            axes = {}
            trends = []
            legend = None
            data_insights = []

            # Extract axes information
            if "axes:" in analysis_text.lower() or "axis:" in analysis_text.lower():
                for line in lines:
                    if "axis" in line.lower() or "-" in line:
                        parts = line.split(":")
                        if len(parts) == 2:
                            axes[parts[0].strip()] = parts[1].strip()

            # Extract trends
            if "trend" in analysis_text.lower():
                for line in lines:
                    if "trend" in line.lower():
                        trend = line.replace("trend:", "").replace("Trend:", "").strip()
                        if trend:
                            trends.append(trend)

            # Extract insights
            if "insight" in analysis_text.lower():
                for line in lines:
                    if "insight" in line.lower():
                        insight = (
                            line.replace("insight:", "")
                            .replace("Insight:", "")
                            .replace("•", "")
                            .replace("-", "")
                            .strip()
                        )
                        if insight:
                            data_insights.append(insight)

            return ChartAnalysis(
                title=title[:100],  # Limit title length
                description=description[:500],  # Limit description
                axes=axes if axes else {},
                trends=trends,
                legend=legend,
                data_insights=data_insights if data_insights else None,
            )

        except Exception as e:
            logger.warning(f"Failed to parse chart analysis: {str(e)}")
            return ChartAnalysis(
                title="Untitled Chart",
                description=analysis_text,
                axes={},
                trends=[],
            )

    async def analyze_multiple_charts(
        self,
        image_list: list[tuple[bytes, Optional[str]]],
    ) -> list[Optional[ChartAnalysis]]:
        """
        Analyze multiple chart images concurrently.

        Args:
            image_list: List of (image_bytes, context) tuples.

        Returns:
            List of ChartAnalysis objects.
        """
        tasks = [
            self.analyze_chart(image_bytes, context)
            for image_bytes, context in image_list
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        analyses = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Chart analysis task failed: {str(result)}")
                analyses.append(None)
            else:
                analyses.append(result)

        return analyses
