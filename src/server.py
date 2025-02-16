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
import sys
from aiohttp import web
from generate import TransformersBackend, process_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logging goes to stdout
)
logger = logging.getLogger(__name__)

class MeshGenerationServer:
    def __init__(self, host='localhost', port=11434):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_post('/api/generate', self.handle_generate)
        self.backend = TransformersBackend()

    @staticmethod
    def create_response_json(content, done=False, error=None):
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

    async def stream_response(self, prompt, temperature, max_tokens, response):
        """Generate mesh and stream response in Ollama-compatible format."""
        try:
            logger.info("Starting mesh generation")

            # Get the stream from the model
            stream = self.backend.generate(
                prompt,
                temperature,
                max_tokens,
                timeout=900.0
            )

            # Stream initial tokens
            for token in [" \n", "```", "obj", "\n"]:
                await response.write(self.create_response_json(token).encode())

            # Process stream using the existing function
            async def process_async():
                for content in process_stream(stream, is_ollama=False, verbose=True):
                    if content:
                        # Send the complete line as one response
                        logger.debug(f"Sending content: {content}")
                        await response.write(self.create_response_json(content).encode())
                        await asyncio.sleep(0.01)  # Small delay between chunks

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

            logger.info("Completed mesh generation")
            return True

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            error_msg = self.create_response_json("", done=True, error=str(e))
            await response.write(error_msg.encode())
            return False

    async def handle_generate(self, request):
        """Handle POST requests to /api/generate endpoint."""
        try:
            logger.info("Received generate request")

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
            logger.info("Response prepared, starting streaming")

            # Generate and stream the mesh
            success = await self.stream_response(prompt, temperature, max_tokens, response)

            if success:
                logger.info("Successfully completed streaming response")
            else:
                logger.error("Failed to complete streaming response")

            await response.write_eof()
            return response

        except Exception as e:
            logger.error(f"Error in handle_generate: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": str(e)}),
                content_type='application/json'
            )

    async def start(self):
        """Start the server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Server started at http://{self.host}:{self.port}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start the LLaMA-Mesh generation server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=11434,
                        help='Port to listen on (default: 11434)')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    server = MeshGenerationServer(args.host, args.port)
    await server.start()

    try:
        # Keep the server running
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main())