from tools.base_tool import BaseTool
import requests

class WeatherTool(BaseTool):
    """Weather information tool"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get current weather information for a location"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "City name (e.g., 'London', 'New York')"
                }
            },
            "required": ["location"]
        }
    
    def execute(self, location: str) -> dict:
        """Get weather data"""
        print(f"🌤️  Fetching weather for: {location}")
        
        if not self.api_key:
            return {
                "location": location,
                "temperature": 22,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "note": "Simulated data. Set OPENWEATHER_API_KEY for real data."
            }
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            return {
                "location": location,
                "temperature": data['main']['temp'],
                "condition": data['weather'][0]['description'],
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed']
            }
        except Exception as e:
            return {"error": f"Weather API error: {str(e)}"}
