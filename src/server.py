#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py

HTTP server wrapper for generate.py that implements an Ollama-compatible API endpoint
"""

import argparse
import asyncio
import datetime
import json
import logging
import logging.handlers
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, DefaultDict

from aiohttp import web
from aiohttp.web import middleware
from generate import TransformersBackend, process_stream

# Configure structured logging
@dataclass
class RequestLogData:
    timestamp: str
    ip: str
    method: str
    path: str
    user_agent: str
    status: int
    duration: float
    error: str = None

class JSONFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, RequestLogData):
            return json.dumps(record.msg.__dict__)
        return super().format(record)

def setup_logging(log_dir: Path):
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure application logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)

    # Configure access logger
    access_logger = logging.getLogger('access')
    access_logger.setLevel(logging.INFO)

    # Rotating file handler for application logs
    app_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    app_logger.addHandler(app_handler)

    # JSON rotating file handler for access logs
    access_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'access.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    access_handler.setFormatter(JSONFormatter())
    access_logger.addHandler(access_handler)

    return app_logger, access_logger

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests: DefaultDict[str, list] = defaultdict(list)

    def is_rate_limited(self, ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[ip] = [req_time for req_time in self.requests[ip]
                             if req_time > minute_ago]

        # Check rate limit
        if len(self.requests[ip]) >= self.requests_per_minute:
            return True

        # Add new request
        self.requests[ip].append(now)
        return False

# Resource management
class ResourceManager:
    def __init__(self, max_concurrent: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_clients: Dict[str, float] = {}

    async def acquire(self, ip: str) -> bool:
        # Clean up stale clients
        now = time.time()
        stale = [ip for ip, timestamp in self.active_clients.items()
                 if now - timestamp > 900]  # 15 minutes timeout
        for stale_ip in stale:
            del self.active_clients[stale_ip]

        if ip in self.active_clients:
            return False

        if await self.semaphore.acquire():
            self.active_clients[ip] = now
            return True
        return False

    def release(self, ip: str):
        if ip in self.active_clients:
            del self.active_clients[ip]
            self.semaphore.release()

class MeshGenerationServer:
    def __init__(self, host='localhost', port=11434, log_dir: Path = Path('/logs')):
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[self.log_middleware])
        self.app.router.add_post('/api/generate', self.handle_generate)
        self.backend = TransformersBackend()

        # Set up logging
        self.logger, self.access_logger = setup_logging(log_dir)

        # Set up rate limiting and resource management
        self.rate_limiter = RateLimiter(requests_per_minute=10)
        self.resource_manager = ResourceManager(max_concurrent=3)

    @middleware
    async def log_middleware(self, request: web.Request, handler):
        start_time = time.time()
        ip = request.headers.get('X-Forwarded-For', request.remote)

        try:
            response = await handler(request)
            duration = time.time() - start_time

            # Log the request
            log_data = RequestLogData(
                timestamp=datetime.datetime.utcnow().isoformat(),
                ip=ip,
                method=request.method,
                path=request.path,
                user_agent=request.headers.get('User-Agent', 'Unknown'),
                status=response.status,
                duration=duration
            )
            self.access_logger.info(log_data)

            return response

        except Exception as e:
            duration = time.time() - start_time
            log_data = RequestLogData(
                timestamp=datetime.datetime.utcnow().isoformat(),
                ip=ip,
                method=request.method,
                path=request.path,
                user_agent=request.headers.get('User-Agent', 'Unknown'),
                status=500,
                duration=duration,
                error=str(e)
            )
            self.access_logger.error(log_data)
            raise

    def create_response_json(self, content, done=False, error=None):
        """Create a properly formatted JSON response."""
        response = {
            "model": "llama-mesh",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "response": content,
            "done": done
        }
        if error:
            response["error"] = error
        if done:
            response["done_reason"] = "stop"
        return json.dumps(response) + "\n"

    async def stream_response(self, prompt, temperature, max_tokens, ip, response):
        try:
            self.logger.info(f"Starting mesh generation for {ip}")

            stream = self.backend.generate(
                prompt,
                temperature,
                max_tokens,
                timeout=900.0
            )

            # Stream initial tokens
            for token in [" \n", "```", "obj", "\n"]:
                await response.write(self.create_response_json(token).encode())

            # Process stream
            async def process_async():
                for content in process_stream(stream, is_ollama=False, verbose=True):
                    if content:
                        await response.write(self.create_response_json(content).encode())
                        await asyncio.sleep(0.01)

            await process_async()

            # Send closing tokens
            await response.write(self.create_response_json("```\n").encode())

            # Send final done message
            final_response = {
                "model": "llama-mesh",
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "response": "",
                "done": True,
                "done_reason": "stop"
            }
            await response.write((json.dumps(final_response) + "\n").encode())

            self.logger.info(f"Completed mesh generation for {ip}")
            return True

        except Exception as e:
            self.logger.error(f"Error in stream_response for {ip}: {e}")
            error_msg = self.create_response_json("", done=True, error=str(e))
            await response.write(error_msg.encode())
            return False

        finally:
            self.resource_manager.release(ip)

    async def handle_generate(self, request: web.Request):
        ip = request.headers.get('X-Forwarded-For', request.remote)

        # Check rate limit
        if self.rate_limiter.is_rate_limited(ip):
            self.logger.warning(f"Rate limit exceeded for {ip}")
            return web.Response(
                status=429,
                text=json.dumps({"error": "Rate limit exceeded"}),
                content_type='application/json'
            )

        # Check resource availability
        if not await self.resource_manager.acquire(ip):
            self.logger.warning(f"Resource limit reached for {ip}")
            return web.Response(
                status=503,
                text=json.dumps({"error": "Server is busy"}),
                content_type='application/json'
            )

        try:
            # Parse request body
            body = await request.json()

            # Extract parameters
            prompt = body.get('prompt')
            options = body.get('options', {})
            temperature = options.get('temperature', 0.95)
            max_tokens = options.get('num_predict', 4096)

            if not prompt:
                return web.Response(
                    status=400,
                    text=json.dumps({"error": "Missing prompt parameter"}),
                    content_type='application/json'
                )

            # Set up streaming response
            response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': 'application/x-ndjson',
                    'Connection': 'keep-alive'
                }
            )
            await response.prepare(request)

            # Generate and stream the mesh
            success = await self.stream_response(
                prompt, temperature, max_tokens, ip, response
            )

            await response.write_eof()
            return response

        except Exception as e:
            self.logger.error(f"Error in handle_generate for {ip}: {e}")
            self.resource_manager.release(ip)
            return web.Response(
                status=500,
                text=json.dumps({"error": str(e)}),
                content_type='application/json'
            )

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.logger.info(f"Server started at http://{self.host}:{self.port}")

def get_default_log_dir():
    """Get default log directory based on environment"""
    if os.path.exists('/.dockerenv'):  # Check if we're in a Docker container
        return Path('/logs')

    # For local development, use a directory in the current working directory
    return Path('./logs')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start the LLaMA-Mesh generation server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=11434,
                        help='Port to listen on (default: 11434)')
    parser.add_argument('--log-dir', type=Path, default=get_default_log_dir(),
                        help='Directory for log files (default: ./logs or /logs in Docker)')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    server = MeshGenerationServer(args.host, args.port, args.log_dir)
    await server.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        server.logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main())