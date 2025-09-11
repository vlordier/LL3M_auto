"""Request reformulation agent for clarifying and improving user requests."""

import asyncio
import json
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..utils.types import AgentResponse, AgentType, WorkflowState
from .base import EnhancedBaseAgent
from .utils import check_user_termination


class Question(BaseModel):
    """Structured question model."""

    id: str
    text: str
    options: list[str] | None = None
    type: str = "choice"


class ReformulationResponse(BaseModel):
    """Structured response model for reformulation."""

    reformulated_request: str
    improvements: str
    questions: list[Question]
    summary: str | None = None
    tags: list[str] | None = None


class RequestReformulationAgent(EnhancedBaseAgent):
    """Agent that reformulates user requests for clarity and completeness."""

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        """Initialize the request reformulation agent."""
        super().__init__(config, client)

        self.SYSTEM_PROMPT = """You are a 3D asset request reformulation specialist.
Your role is to take user requests for 3D assets and reformulate them into clear,
specific, and actionable descriptions.

Your task is to:
1. Analyze the user's original request
            2. Identify any ambiguities or missing details
3. Reformulate the request to be more specific and clear
4. Generate structured clarifying questions with options when possible
5. Create relevant tags and a summary

Focus on these aspects:
- **Object Type**: What exactly should be created?
- **Style/Aesthetics**: What visual style (realistic, cartoon, abstract, etc.)?
- **Materials/Textures**: What surfaces and materials?
- **Scale/Size**: Relative or absolute dimensions
- **Level of Detail**: How detailed should it be?
- **Context/Purpose**: How will it be used?

For questions:
- Provide multiple choice options when possible
- Make questions specific and actionable
- Focus on aspects that significantly impact the final 3D model
- Limit to 3-4 most important questions

For tags: Include relevant categories like style, material, object type, etc.

Example structured output:
{
  "reformulated_request": "Create a modern office chair with ergonomic design...",
  "improvements": "Added specific style, materials, and intended use context",
  "questions": [
    {
      "id": "style_preference",
      "text": "What specific design style do you prefer?",
      "options": ["Modern minimalist", "Traditional executive", "Gaming chair", "Scandinavian"],
      "type": "choice"
    }
  ],
  "summary": "Office chair with modern ergonomic features",
  "tags": ["furniture", "office", "modern", "ergonomic"]
}"""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.REFORMULATION

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process the user request and generate a reformulated version."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Check for user termination request
            if check_user_termination(state):
                self.logger.info("User requested workflow termination")
                state.should_continue = False
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    data="",
                    message="Workflow terminated by user request",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            if not state.prompt or not state.prompt.strip():
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="No user prompt provided for reformulation",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            self.logger.info("Starting request reformulation", prompt_length=len(state.prompt))

            # Prepare the reformulation prompt
            user_prompt = f"""Please reformulate this 3D asset request using structured output:

Original Request: "{state.prompt}"

Analyze the request and provide:
1. A detailed, actionable reformulated request
2. Explanation of improvements made
3. Structured questions with multiple choice options when possible
4. A brief summary
5. Relevant tags

Make the reformulated request specific enough for 3D modeling while preserving creative intent."""

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # Try structured output first
            structured_response = await self._make_structured_request(messages)

            if structured_response:
                # Use structured response
                reformulated_request = structured_response.reformulated_request
                improvements = structured_response.improvements
                questions = structured_response.questions
                summary = structured_response.summary
                tags = structured_response.tags
                reformulation_response = json.dumps(structured_response.model_dump(), indent=2)
            else:
                # Fallback to traditional text response
                self.logger.warning("Structured output unavailable; falling back to text response")
                reformulation_response = await self.make_openai_request(messages)

                reformulated_request = self._extract_reformulated_request(reformulation_response)
                improvements = self._extract_improvements(reformulation_response)
                questions_text = self._extract_questions(reformulation_response)
                questions = []  # Will be text format
                summary = None
                tags = None

            if not reformulation_response.strip():
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="Failed to generate reformulation - empty response",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            # Store reformulation data in state for later use
            if not hasattr(state, "reformulation_data"):
                state.reformulation_data = {}

            # Convert Pydantic Question objects to dictionaries for JSON serialization
            questions_data: str | list[dict[str, Any]]
            if isinstance(questions, list) and questions and hasattr(questions[0], "model_dump"):
                # Structured response with Question objects
                questions_data = [q.model_dump() for q in questions]
            elif "questions_text" in locals():
                # Fallback text response
                questions_data = questions_text
            else:
                # Empty list from fallback or other cases
                questions_data = ""

            state.reformulation_data = {
                "original_prompt": state.prompt,
                "reformulated_request": reformulated_request,
                "improvements": improvements,
                "questions": questions_data,
                "summary": summary,
                "tags": tags,
                "full_response": reformulation_response,
                "needs_confirmation": True,  # Updated key name for consistency
                "structured_output": isinstance(questions, list) and len(questions) > 0,
            }

            # Set flag to indicate the workflow should wait for user confirmation
            state.awaiting_user_confirmation = True

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Request reformulation completed",
                reformulated_length=len(reformulated_request),
                has_questions=bool(questions)
                if isinstance(questions, list)
                else bool(questions.strip() if questions else ""),
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=reformulation_response,
                message=f"Reformulated request with {len(improvements.split())} improvements",
                execution_time=execution_time,
                metadata={
                    "original_prompt": state.prompt,
                    "reformulated_request": reformulated_request,
                    "improvements": improvements,
                    "questions": questions_data,
                    "summary": summary,
                    "tags": tags,
                    "needs_confirmation": True,
                    "structured_output": isinstance(questions, list) and len(questions) > 0,
                },
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.exception("Request reformulation failed", error=str(e))

            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data="",
                message=f"Reformulation failed: {str(e)}",
                execution_time=execution_time,
            )

    def _extract_reformulated_request(self, response: str) -> str:
        """Extract the reformulated request from the response."""
        lines = response.split("\n")
        reformulated_lines = []
        collecting = False

        for line in lines:
            if "**Reformulated Request:**" in line or "Reformulated Request:" in line:
                collecting = True
                continue
            if (
                "**Improvements Made:**" in line
                or "Improvements Made:" in line
                or "**Questions" in line
                or "Questions:" in line
            ):
                collecting = False
                break
            if collecting and line.strip():
                reformulated_lines.append(line.strip())

        return " ".join(reformulated_lines) if reformulated_lines else response.split("\n")[0]

    def _extract_improvements(self, response: str) -> str:
        """Extract the improvements section from the response."""
        lines = response.split("\n")
        improvement_lines = []
        collecting = False

        for line in lines:
            if "**Improvements Made:**" in line or "Improvements Made:" in line:
                collecting = True
                continue
            if "**Questions" in line or "Questions:" in line or "**" in line and collecting:
                collecting = False
                break
            if collecting and line.strip():
                improvement_lines.append(line.strip())

        return " ".join(improvement_lines) if improvement_lines else "Request was clarified and made more specific."

    def _extract_questions(self, response: str) -> str:
        """Extract any questions from the response."""
        lines = response.split("\n")
        question_lines = []
        collecting = False

        for line in lines:
            if "**Questions" in line or "Questions:" in line:
                collecting = True
                continue
            if collecting and line.strip():
                if line.strip().startswith("**") and not line.strip().startswith("**Questions"):
                    break
                question_lines.append(line.strip())

        return " ".join(question_lines) if question_lines else ""

    async def _make_structured_request(
        self, messages: list[ChatCompletionMessageParam], max_retries: int = 3
    ) -> ReformulationResponse | None:
        """Make a structured OpenAI request with retry logic and response parsing."""
        if not hasattr(self, "client") or not self.client:
            self.logger.warning("No OpenAI client available for structured request")
            return None

        # Get the underlying OpenAI client
        openai_client = getattr(self.client, "client", None)
        if not openai_client:
            self.logger.warning("Cannot access underlying OpenAI client for structured output")
            return None

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Making structured request attempt {attempt + 1}/{max_retries}")

                # Use OpenAI's structured output with response_format
                # Note: Structured outputs only work with specific models like gpt-4o, gpt-4o-mini
                structured_model = "gpt-4o"  # Force use of structured output compatible model
                response = await openai_client.chat.completions.create(
                    model=structured_model,
                    messages=messages,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 2000),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "reformulation_response",
                            "schema": ReformulationResponse.model_json_schema(),
                        },
                    },
                )

                content = response.choices[0].message.content
                if not content:
                    continue

                # Parse the structured JSON response
                try:
                    parsed_data = json.loads(content)
                    structured_response = ReformulationResponse(**parsed_data)

                    # Validate required fields
                    if not (
                        structured_response.reformulated_request.strip() and structured_response.improvements.strip()
                    ):
                        continue

                    self.logger.info(f"Structured request successful on attempt {attempt + 1}")
                    return structured_response

                except (json.JSONDecodeError, ValueError) as parse_error:
                    self.logger.warning(f"Failed to parse structured response on attempt {attempt + 1}: {parse_error}")
                    continue

            except Exception as e:
                self.logger.warning(f"Structured request attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + (0.1 * attempt)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.exception(f"All {max_retries} structured request attempts failed")
                    return None

        return None

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate that the input state has a prompt to reformulate."""
        if not state.prompt or not state.prompt.strip():
            self.logger.error("No prompt provided for reformulation")
            return False
        return True
