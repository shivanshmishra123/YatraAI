# 🔑 API Keys Guide for Yatra AI

Yatra AI is designed with a **zero-failure architecture**. It functions completely in **Local Domestic Engine Mode** with zero external API keys. However, configuring API keys unlocks live web querying, real-time IRCTC berth status, Google Maps traffic data, and live LLM generations.

---

## 📋 Overview of API Keys

| Service | Environment Variable | Required? | Purpose | Free Tier / Pricing |
|:---|:---|:---:|:---|:---|
| **Groq Cloud** (Recommended) | `GROQ_API_KEY` | Optional | Powers **Llama 3.3 70B Versatile** for lightning-fast NLU extraction & ReAct agent | **100% Free Tier** (30 RPM, 14,400 req/day, zero credit card) |
| **Google AI Studio** | `GOOGLE_API_KEY` | Optional | Powers **Gemini 1.5 Pro** for deep multi-city route optimization & budget survival | Free tier available (15 RPM) |
| **Google Maps** | `GMAPS_API_KEY` | Optional | Real-time driving directions, traffic, place search, and hotel data | $200 free monthly credit (~40,000 requests) |
| **WeatherAPI / OpenWeather**| `WEATHER_API_KEY` | Optional | Live regional 7-day weather forecasts and storm/heatwave alerts | Free tier: 1,000 calls/day |
| **Amadeus Travel** | `AMADEUS_API_KEY`<br>`AMADEUS_API_SECRET` | Optional | Live domestic flight search (DEL, BOM, BLR, MAA, CCU, GOI) | Free Self-Service tier: 2,000 calls/month |
| **OpenRouter / OpenAI** | `OPENROUTER_API_KEY`<br>`OPENAI_API_KEY` | Optional | Alternative free models or legacy OpenAI endpoints | Free models available on OpenRouter |
| **LangSmith** | `LANGCHAIN_API_KEY` | Optional | Observability, execution traces, and agent latency profiling | Free personal tier |

---

## 🛠️ Step-by-Step Acquisition Guide

### 1. Groq Cloud Free API Key (`GROQ_API_KEY`) — Recommended
Groq provides **ultra-fast inference completely for free** without requiring a credit card:
1. Visit **[console.groq.com](https://console.groq.com/)**.
2. Sign in with your Google or GitHub account.
3. In the left navigation, click **API Keys**.
4. Click **Create API Key**, name it `yatra-ai`, and copy the key (`gsk_...`).
5. Add to your `.env`:
   ```bash
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```
*Free Tier limits: 30 requests/minute, 14,400 requests/day, Llama 3.3 70B with 128k context.*

---

### 2. Google AI Studio Key (`GOOGLE_API_KEY`)
1. Visit [aistudio.google.com](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** in the top navigation.
4. Select or create a Google Cloud project, and click **Create API key in existing project**.
5. Copy the generated key (`AIza...`) and add to `.env`:
   ```bash
   GOOGLE_API_KEY=AIzaSyYourGoogleKeyHere
   ```

---

### 3. Google Maps Platform (`GMAPS_API_KEY`)
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the following APIs:
   - **Directions API**
   - **Distance Matrix API**
   - **Places API (New)**
3. Navigate to **Credentials** ➔ **Create Credentials** ➔ **API Key**.
4. Restrict the key to the enabled APIs for security.
5. Add to `.env`:
   ```bash
   GMAPS_API_KEY=AIzaSyYourGmapsKeyHere
   ```

---

### 4. Weather API (`WEATHER_API_KEY`)
1. Visit [weatherapi.com](https://www.weatherapi.com/) or [openweathermap.org](https://openweathermap.org/api).
2. Sign up for a free account.
3. Locate your API Key under your dashboard profile.
4. Add to `.env`:
   ```bash
   WEATHER_API_KEY=your_weather_api_key_here
   ```

---

### 5. Amadeus Flight API (`AMADEUS_API_KEY` & `AMADEUS_API_SECRET`)
1. Register at [developers.amadeus.com](https://developers.amadeus.com/).
2. Click **My Self-Service Workspace** ➔ **Create New App**.
3. Copy both the **API Key** (Client ID) and **API Secret** (Client Secret).
4. Add to `.env`:
   ```bash
   AMADEUS_API_KEY=your_amadeus_client_id
   AMADEUS_API_SECRET=your_amadeus_client_secret
   ```

---

## 🔒 Security Best Practices
- Never commit your `.env` file to version control. The `.gitignore` file is configured to exclude all `.env` files automatically.
- To share environment requirements with collaborators, update [.env.example](file:///c:/Users/13shi/OneDrive/Desktop/traverlplanner/.env.example) with placeholder values.
