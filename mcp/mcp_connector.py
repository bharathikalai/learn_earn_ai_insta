from mcp_client_for_ollama.client import MCPClient
import asyncio

async def main():
    client = MCPClient(
        model="qwen2.5:1.5b",
        host="http://localhost:11434"
    )
    
    await client.connect_to_servers(
        server_urls=["http://127.0.0.1:8000/mcp"]
    )
    
    # Disable human-in-the-loop if the attribute exists
    if hasattr(client, 'human_in_loop'):
        client.human_in_loop = False
    
    await client.chat_loop()

if __name__ == "__main__":
    asyncio.run(main())