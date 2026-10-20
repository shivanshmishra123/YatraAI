"""Health check utility for Yatra AI MCP servers."""

import asyncio
import json
import os
import sys
from pathlib import Path
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG_PATH = os.getenv(
    "MCP_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "yatra_agent" / "config" / "servers_config.json")
)


async def check_server(server: dict) -> dict:
    name = server.get("name", "Unknown")
    url = server.get("url", "")
    required = server.get("required", False)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url.replace("/sse", "/health") if "/sse" in url else url)
            status = "✅ ONLINE" if resp.status_code in (200, 404, 405) else f"⚠️ HTTP {resp.status_code}"
    except Exception as e:
        status = "❌ OFFLINE (Mock/Fallback available)"

    return {
        "name": name,
        "url": url,
        "required": required,
        "status": status,
    }


async def main():
    print("=" * 60)
    print("🌍 Yatra AI — MCP Servers Health Check")
    print("=" * 60)

    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Configuration not found at: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    servers = config.get("mcp_servers", [])
    print(f"Found {len(servers)} configured MCP servers in {CONFIG_PATH}:\n")

    tasks = [check_server(s) for s in servers]
    results = await asyncio.gather(*tasks)

    for res in results:
        req_tag = "[Required]" if res["required"] else "[Optional]"
        print(f"• {res['name']} {req_tag}: {res['status']}")
        print(f"  Endpoint: {res['url']}")

    print("\n" + "=" * 60)
    print("Health check completed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
