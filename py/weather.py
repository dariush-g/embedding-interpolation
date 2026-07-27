
import requests
import sys
import json
from llama_cpp import Llama
import re

llm = Llama(
    model_path="/Users/dariushghassemieh/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f",
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False, 
)

MAX_STEPS = 5

# WMO weather codes -> human text (subset; Open-Meteo returns these numbers)
WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}

def get_weather(city: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=30,
    ).json()
    if not geo.get("results"):
        return f"Could not find a location named {city!r}."
    loc = geo["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]
    place = f"{loc['name']}, {loc.get('country', '')}".strip(", ")
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        }, timeout=30,
    ).json()["current"]
    desc = WMO.get(wx["weather_code"], "unknown conditions")
    return (f"{place}: {desc}, {wx['temperature_2m']}°C, "
            f"humidity {wx['relative_humidity_2m']}%, "
            f"wind {wx['wind_speed_10m']} km/h.")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use this whenever the "
                       "user asks about weather, temperature, or what to wear outside.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Reykjavik' or 'Tokyo'.",
                },
            },
            "required": ["city"],
        },
    },
}]

TOOL_FUNCS = {"get_weather" : get_weather}

def chat_once(messages):
    out = llm.create_chat_completion(messages=messages, tools=TOOLS)
    return out["choices"][0]["message"]

def run(query):
    messages = [{"role": "user", "content": query}]

    for _ in range(MAX_STEPS):
        msg = chat_once(messages)
        content = msg.get("content") or ""

        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        if not m:                                  # no tool call -> final answer
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        call = json.loads(m.group(1))
        name, args = call["name"], call["arguments"]
        print(f"  [model called {name}({args})]")
        result = TOOL_FUNCS[name](**args) if name in TOOL_FUNCS else f"Error: no tool named {name}."
        print(f"  [tool returned: {result}]")

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user",
                         "content": f"<tool_response>\n{result}\n</tool_response>"})

    return "Stopped: hit the step limit."

def main():
    if len(sys.argv) > 1:
        print(run(" ".join(sys.argv[1:])))
        return
    print("Ask about the weather. Empty line or Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print(run(q))
 
 
if __name__ == "__main__":
    main()