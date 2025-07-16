# Quant Peek

## Project Overview

**Quant Peek** is a full-stack stock analysis tool that combines backend data processing with frontend visualization, including a web interface.

### What the Project Does:

* **Backend (Python)**

  * Takes user input for stock symbol, start date, and end date.
  * Fetches historical stock price data from Yahoo Finance using `yfinance`.
  * Calculates the RSI (Relative Strength Index) using `numpy`.
  * Provides processed data for frontend use.

* **Frontend (Web Visualization)**

  * Developed with **TypeScript** and **JavaScript**.
  * Visualizes stock data and RSI through interactive web components.
  * Allows users to view charts directly from the browser via **HTTPS**.
  * Live Demo: https://quant-peek-iltf.vercel.app/

* **Desktop Visualization (Optional CLI View)**

  * Uses `matplotlib` to generate:

    * A line chart of the stock's closing prices.
    * An RSI chart.
  * Displays the charts in pop-up windows when run via CLI.

This setup enables both web-based and desktop-based stock data analysis.

---

## Technologies Used

| Technology     | Purpose                          |
| -------------- | -------------------------------- |
| **Python**     | Backend logic                    |
| **TypeScript** | Web frontend logic               |
| **JavaScript** | Web frontend interaction         |
| **yfinance**   | Fetching financial data          |
| **pandas**     | Data processing and analysis     |
| **numpy**      | RSI calculation                  |
| **matplotlib** | Data visualization (Desktop CLI) |

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Busedkc/quant-peek.git
cd quant-peek
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend (CLI Option)

```bash
python quant_peek.py
```

#### Provide Input in Terminal

* Stock symbol (e.g., `AAPL`)
* Start date (e.g., `2024-01-01`)
* End date (e.g., `2024-07-01`)

Once the data is entered, the charts will be displayed automatically via desktop.

### 4. Run Frontend (Web Interface)

Navigate to the `web` directory and install frontend dependencies:

```bash
cd web
npm install
npm start
```

This will start the local development server. The web visualization interface can be accessed securely via **HTTPS** (e.g., `https://localhost:3000`).

---

This README covers only the features and usage present in the current repository.
