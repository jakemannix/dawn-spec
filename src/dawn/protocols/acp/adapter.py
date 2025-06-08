"""
AGNTCY ACP Protocol Adapter

This module provides the adapter for integrating Dawn agents with
the AGNTCY Agent Connect Protocol (ACP).
"""

from typing import Any, Dict, Optional
from aiohttp import web
from agntcy_acp import AsyncACPClient, ApiClientConfiguration

from dawn.core.agent import Agent
from dawn.protocols.base import ProtocolAdapter, ProtocolEndpoint
from dawn.utils.logging import get_logger, LogEvent


class ACPAdapter(ProtocolAdapter):
    """
    AGNTCY ACP protocol adapter.
    
    This adapter exposes Dawn agents via the ACP protocol, providing
    HTTP endpoints for agent discovery and interaction.
    """
    
    def __init__(self, logger=None):
        """Initialize the ACP adapter."""
        super().__init__(logger)
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
    
    def get_protocol_name(self) -> str:
        """Get the protocol name."""
        return "acp"
    
    def get_protocol_version(self) -> str:
        """Get the protocol version."""
        return "1.0"
    
    async def start(self, agent: Agent, host: str = "localhost", port: int = 8080) -> ProtocolEndpoint:
        """
        Start the ACP server for an agent.
        
        Args:
            agent: The Dawn agent to expose
            host: Host to bind to
            port: Port to bind to
            
        Returns:
            ProtocolEndpoint with server information
        """
        if self._is_running:
            raise RuntimeError("ACP adapter is already running")
        
        self.agent = agent
        
        # Create the web application
        self.app = await self._create_app(agent)
        
        # Start the server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        
        # Create endpoint info
        self.endpoint = ProtocolEndpoint(
            protocol="acp",
            host=host,
            port=port,
            base_url=f"http://{host}:{port}",
            metadata={
                "version": self.get_protocol_version(),
                "agent_id": agent.agent_id,
            }
        )
        
        self._is_running = True
        
        self.logger.info(
            f"Started ACP server for {agent.name} at {self.endpoint.url}"
        )
        
        return self.endpoint
    
    async def stop(self) -> None:
        """Stop the ACP server."""
        if not self._is_running:
            return
        
        self.logger.info("Stopping ACP server")
        
        if self.site:
            await self.site.stop()
            self.site = None
        
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
        
        self.app = None
        self._is_running = False
        self.endpoint = None
        
        self.logger.info("ACP server stopped")
    
    async def _create_app(self, agent: Agent) -> web.Application:
        """
        Create the aiohttp application with ACP endpoints.
        
        Args:
            agent: The Dawn agent to expose
            
        Returns:
            Configured aiohttp Application
        """
        app = web.Application()
        
        # Add routes
        app.router.add_get("/", self._handle_root)
        app.router.add_get("/agent.json", self._handle_agent_card)
        app.router.add_get("/.well-known/agent.json", self._handle_agent_card)
        app.router.add_post("/chat", self._handle_chat)
        app.router.add_get("/health", self._handle_health)
        
        # Store agent reference
        app["agent"] = agent
        
        return app
    
    async def _handle_root(self, request: web.Request) -> web.Response:
        """Handle root endpoint."""
        agent = request.app["agent"]
        return web.json_response({
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "protocol": "acp",
            "endpoints": {
                "agent_card": "/agent.json",
                "chat": "/chat",
                "health": "/health"
            }
        })
    
    async def _handle_agent_card(self, request: web.Request) -> web.Response:
        """Handle agent card request."""
        agent = request.app["agent"]
        card = self.get_agent_card_for_protocol()
        
        # Add ACP-specific fields
        card["acp"] = {
            "endpoint": str(request.url.with_path("/")),
            "protocol_version": self.get_protocol_version(),
            "chat_endpoint": str(request.url.with_path("/chat")),
        }
        
        return web.json_response(card)
    
    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle chat request."""
        agent = request.app["agent"]
        
        try:
            # Parse request
            data = await request.json()
            message = data.get("message", "")
            context = data.get("context", {})
            
            if not message:
                return web.json_response(
                    {"error": "No message provided", "success": False},
                    status=400
                )
            
            # Process with agent
            result = await agent.process_message(message, context)
            
            # Format response
            response = {
                "response": result.get("response", ""),
                "success": result.get("success", True),
                "metadata": {
                    "agent_id": agent.agent_id,
                    "tools_used": result.get("tools_used", []),
                    "reasoning_trace": result.get("reasoning_trace", []),
                }
            }
            
            return web.json_response(response)
            
        except Exception as e:
            self.logger.error(f"Error handling chat request: {e}", exception=e)
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check."""
        agent = request.app["agent"]
        return web.json_response({
            "status": "healthy",
            "agent": agent.name,
            "protocol": "acp",
            "version": self.get_protocol_version()
        }) 