# 🌱 Crop Price Prediction System

A hybrid ARIMA-LSTM model that forecasts agricultural commodity prices with 7-day predictions.



## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## ✨ Features
- **7-day price forecasts** for 25+ crops
- **Web dashboard** with historical data visualization
- **Price alert system** (email/SMS notifications)
- **Regional price comparisons**
- **Hybrid ARIMA-LSTM** model for accurate predictions

## 🛠 Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)
- Download all the required datasets from website called [Agmarkent](http://localhost:5000)
- Weather datas from [Meteostat](https://meteostat.net/en/)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crop-price-predictor.git
cd crop-price-predictor
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🖥 Usage

### 1. Generate Models (First-Time Setup)
```bash
python model.py
```
*This will:*
- Train ARIMA and LSTM models
- Save `.pkl` and `.h5` files in `/models`
- Takes 10-30 minutes depending on hardware

### 2. Start the Web Application
```bash
python app.py
```
Access the dashboard at: [http://localhost:5000](http://localhost:5000)

### 3. Using the System
1. Select a crop from the dropdown
2. View 7-day price forecasts
3. (Optional) Set price alerts via the web interface

## 📂 Project Structure
```
crop-price-predictor/
├── app.py                # Flask web server
├── model.py              # Model training code
├── web.html              # Frontend template
├── requirements.txt      # Python dependencies
├── models/               # Auto-generated model storage
├── static/               # CSS/JS assets (if any)
└── sample_data/          # Example datasets
    ├── potatoprice.csv
    └── tomatoprice.csv
```

## 🚨 Troubleshooting
**Q:** Getting "Model files not found" error?  
**A:** Run `python model.py` first to generate required models

**Q:** Flask server not starting?  
**A:** Check port 5000 isn't in use: `lsof -i:5000` (Linux/Mac) or `netstat -ano | findstr 5000` (Windows)

**Q:** Missing dependencies?  
**A:** Reinstall with: `pip install -r requirements.txt`

## 📜 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

### How to Customize:
1. Replace `your-username` with your GitHub username
2. Add actual screenshots (upload to `/images` folder)
3. Update the crop list if you've added/removed any
4. Add your contact info in a "Support" section

### Pro Tip:
Use [Markdown Guide](https://www.markdownguide.org/) for advanced formatting like:
```markdown
![Dashboard Demo](images/screenshot.png)
```

This README provides:
- Clear setup instructions
- Visual structure
- Self-documenting troubleshooting
- Professional presentation

Would you like me to add any specific sections like:
- API documentation?
- Contribution guidelines?
- Deployment instructions?
