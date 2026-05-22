import asyncio
import httpx
import json

async def test_conn():
    api_url = "http://192.168.1.22:2090/api/Order/listorder"
    print(f"Connecting to {api_url} with verify=True...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url)
            print("Status Code:", response.status_code)
            print("Response Headers:", response.headers)
            print("First 200 chars of response:", response.text[:200])
    except httpx.ConnectError as ce:
        print("Connection Error:", ce)
    except httpx.ConnectTimeout as ct:
        print("Connection Timeout:", ct)
    except httpx.HTTPStatusError as hse:
        print("HTTP Status Error:", hse)
    except Exception as e:
        print("Unexpected Error:", type(e), e)

    print("\nConnecting to {api_url} with verify=False (bypass SSL checks) and NO authorization headers...")
    try:
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            response = await client.get(api_url)
            print("Status Code (verify=False):", response.status_code)
            print("First 200 chars of response (verify=False):", response.text[:200])
    except Exception as e:
        print("Error with verify=False:", type(e), e)

asyncio.run(test_conn())
