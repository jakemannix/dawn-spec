# src/implementations/text_matching_agent.py

from typing import Dict, Any
import re
from ..agent_core import MCPIntegratedAgent, AgentContext, AgentResponse, AgentImplementationType


class TextMatchingAgent(MCPIntegratedAgent):
    """
    Text-matching agent implementation that uses string patterns to detect intent.
    This preserves the original non-LLM intent detection logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.intent_patterns = config.get('intent_patterns', self._default_patterns())
        self.default_responses = config.get('default_responses', self._default_responses())
    
    @property
    def implementation_type(self) -> AgentImplementationType:
        return AgentImplementationType.TEXT_MATCHING
    
    async def initialize(self) -> None:
        """Initialize the text matching agent"""
        self.logger.info("Initializing TextMatchingAgent")
        # No special initialization needed for text matching
        
    async def shutdown(self) -> None:
        """Shutdown the text matching agent"""
        self.logger.info("Shutting down TextMatchingAgent")
        # No cleanup needed
    
    async def process_request(self, context: AgentContext) -> AgentResponse:
        """Process request using text pattern matching"""
        user_message = context.user_message.lower().strip()
        
        # Detect intent using pattern matching
        detected_intent = self._detect_intent(user_message)
        self.logger.info(f"Detected intent: {detected_intent}")
        
        # Execute based on detected intent
        if detected_intent == "github_search":
            return await self._handle_github_search(context, user_message)
        elif detected_intent == "arxiv_search":
            return await self._handle_arxiv_search(context, user_message)
        elif detected_intent == "synthesis":
            return await self._handle_synthesis(context, user_message)
        elif detected_intent == "help":
            return await self._handle_help(context)
        else:
            return await self._handle_unknown(context, user_message)
    
    def _detect_intent(self, message: str) -> str:
        """Detect intent using pattern matching"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        return "unknown"
    
    async def _handle_github_search(self, context: AgentContext, message: str) -> AgentResponse:
        """Handle GitHub search intent"""
        # Extract search query from message
        query = self._extract_search_query(message, ["github", "search", "repository", "repo"])
        
        try:
            # Use MCP tool for GitHub search
            result = await self.call_mcp_tool(context, "github_search", {"query": query})
            
            return AgentResponse(
                response_text=f"Found GitHub repositories for '{query}': {result}",
                tools_used=["github_search"],
                reasoning_trace=[
                    f"Detected GitHub search intent",
                    f"Extracted query: {query}", 
                    f"Called github_search tool"
                ],
                confidence_score=0.9
            )
        except Exception as e:
            # Fallback to A2A skill
            try:
                a2a_result = await self.invoke_a2a_skill(
                    context, 
                    "research.github.search.github_search",
                    {"query": query}
                )
                return AgentResponse(
                    response_text=f"GitHub search via A2A: {a2a_result}",
                    skills_invoked=["research.github.search.github_search"],
                    reasoning_trace=[
                        f"MCP tool failed: {e}",
                        f"Fallback to A2A skill successful"
                    ],
                    confidence_score=0.7
                )
            except Exception as e2:
                return AgentResponse(
                    response_text=f"GitHub search failed: {e2}",
                    reasoning_trace=[f"Both MCP and A2A failed"],
                    confidence_score=0.1
                )
    
    async def _handle_arxiv_search(self, context: AgentContext, message: str) -> AgentResponse:
        """Handle arXiv search intent"""
        query = self._extract_search_query(message, ["arxiv", "paper", "research", "academic"])
        
        try:
            # Use MCP tool for arXiv search
            result = await self.call_mcp_tool(context, "arxiv_search", {"query": query})
            
            return AgentResponse(
                response_text=f"Found arXiv papers for '{query}': {result}",
                tools_used=["arxiv_search"],
                reasoning_trace=[
                    f"Detected arXiv search intent",
                    f"Extracted query: {query}",
                    f"Called arxiv_search tool"
                ],
                confidence_score=0.9
            )
        except Exception as e:
            # Fallback to A2A skill
            try:
                a2a_result = await self.invoke_a2a_skill(
                    context,
                    "research.arxiv.search.arxiv_search", 
                    {"query": query}
                )
                return AgentResponse(
                    response_text=f"arXiv search via A2A: {a2a_result}",
                    skills_invoked=["research.arxiv.search.arxiv_search"],
                    reasoning_trace=[
                        f"MCP tool failed: {e}",
                        f"Fallback to A2A skill successful"
                    ],
                    confidence_score=0.7
                )
            except Exception as e2:
                return AgentResponse(
                    response_text=f"arXiv search failed: {e2}",
                    reasoning_trace=[f"Both MCP and A2A failed"],
                    confidence_score=0.1
                )
    
    async def _handle_synthesis(self, context: AgentContext, message: str) -> AgentResponse:
        """Handle synthesis intent"""
        try:
            # Use synthesis A2A skill
            result = await self.invoke_a2a_skill(
                context,
                "research.synthesis.synthesize",
                {"input": message}
            )
            
            return AgentResponse(
                response_text=f"Synthesis result: {result}",
                skills_invoked=["research.synthesis.synthesize"],
                reasoning_trace=[
                    f"Detected synthesis intent",
                    f"Called synthesis A2A skill"
                ],
                confidence_score=0.8
            )
        except Exception as e:
            return AgentResponse(
                response_text=f"Synthesis failed: {e}",
                reasoning_trace=[f"Synthesis skill failed"],
                confidence_score=0.1
            )
    
    async def _handle_help(self, context: AgentContext) -> AgentResponse:
        """Handle help intent"""
        help_text = """
Available commands:
- Search GitHub: "search github for [query]" or "find repository [query]"
- Search arXiv: "search arxiv for [query]" or "find papers about [query]"  
- Synthesize: "synthesize [topic]" or "combine research on [topic]"
- Help: "help" or "what can you do"

Available tools: {tools}
Available skills: {skills}
""".format(
            tools=", ".join(context.agent_capabilities.get("tools", [])),
            skills=", ".join(context.agent_capabilities.get("skills", []))
        )
        
        return AgentResponse(
            response_text=help_text,
            reasoning_trace=["Provided help information"],
            confidence_score=1.0
        )
    
    async def _handle_unknown(self, context: AgentContext, message: str) -> AgentResponse:
        """Handle unknown intent"""
        return AgentResponse(
            response_text=f"I didn't understand '{message}'. Type 'help' for available commands.",
            reasoning_trace=[f"No intent pattern matched for: {message}"],
            confidence_score=0.0
        )
    
    def _extract_search_query(self, message: str, keywords: list) -> str:
        """Extract search query from message by removing keywords"""
        query = message
        for keyword in keywords:
            query = re.sub(rf'\b{keyword}\b', '', query, flags=re.IGNORECASE)
        
        # Clean up extra spaces and common words
        query = re.sub(r'\b(for|about|on|search|find)\b', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query if query else "general"
    
    def _default_patterns(self) -> Dict[str, list]:
        """Default intent patterns for text matching"""
        return {
            "github_search": [
                r"search\s+github",
                r"find\s+repository",
                r"github\s+.*search",
                r"look\s+for\s+.*repo",
                r"@github"
            ],
            "arxiv_search": [
                r"search\s+arxiv",
                r"find\s+papers?",
                r"arxiv\s+.*search",
                r"academic\s+.*search",
                r"research\s+papers?",
                r"@arxiv"
            ],
            "synthesis": [
                r"synthesize",
                r"combine\s+research",
                r"merge\s+.*results",
                r"@synthesis"
            ],
            "help": [
                r"^help$",
                r"what\s+can\s+you\s+do",
                r"how\s+do\s+I",
                r"commands?",
                r"usage"
            ]
        }
    
    def _default_responses(self) -> Dict[str, str]:
        """Default responses for common cases"""
        return {
            "greeting": "Hello! I can help you search GitHub repositories, arXiv papers, and synthesize research. What would you like to do?",
            "goodbye": "Goodbye! Feel free to ask if you need help with research later.",
            "error": "I encountered an error processing your request. Please try again or ask for help."
        } 