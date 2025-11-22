from mcp_client_for_ollama.client import MCPClient

client = MCPClient()

# Show all available methods (excluding private ones)
print("Available methods:")
methods = [m for m in dir(client) if not m.startswith('_')]
for method in methods:
    print(f"  - {method}")

print("\n" + "="*50)
print("\nFull help for MCPClient:")
print("="*50)
help(MCPClient)

print("\n" + "="*50)
print("\nFull help for client instance:")
print("="*50)
help(client)