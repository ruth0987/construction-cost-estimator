# Construction Cost Estimator

An AI-powered construction cost estimation application built with Streamlit, LangChain, and Groq. This app helps you estimate construction costs by gathering project details through conversation and generating comprehensive cost estimates.

## Features

- **Interactive Chat Interface**: Natural conversation with AI to gather project scope details
- **Cost Estimation**: Generate detailed budget ranges and bill of quantities
- **Scenario Analysis**: Explore alternative scenarios and their cost impacts
- **Benchmark Comparisons**: Compare estimates with similar projects
- **Vector Database**: Uses ChromaDB for RAG-based cost estimation

## Tech Stack

- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration framework
- **Groq**: Fast LLM inference API
- **ChromaDB**: Vector database for RAG
- **HuggingFace**: Embeddings model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ruth0987/construction-cost-estimator.git
cd construction-cost-estimator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.streamlit/secrets.toml` file (for local development)
   - Add your Groq API key:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```

## Running Locally

```bash
streamlit run streamlit_app.py
```

## Deployment to Streamlit Cloud

1. **Push your code to GitHub** (already done if you're reading this)

2. **Get a Groq API Key**:
   - Sign up at [https://console.groq.com/](https://console.groq.com/)
   - Create an API key

3. **Deploy on Streamlit Cloud**:
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Click "New app"
   - Connect your GitHub repository: `ruth0987/construction-cost-estimator`
   - Set the main file path to: `streamlit_app.py`
   - Click "Advanced settings" and add your secrets:
     ```
     GROQ_API_KEY = "your-groq-api-key-here"
     ```
   - Click "Deploy"

4. **Configure Secrets in Streamlit Cloud**:
   - In your Streamlit Cloud dashboard, go to your app settings
   - Click on "Secrets" tab
   - Add the following:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```

## Usage

1. Start a conversation about your construction project
2. Provide project details such as:
   - Project type (Office, Residential, etc.)
   - Total floor area (sqm)
   - Number of stories
   - Location
   - Key features and specifications
3. Click "Generate Cost Estimate" when you have enough information
4. View the detailed cost estimate, bill of quantities, and scenario analysis
5. Download the estimate as JSON

## Project Structure

```
construction-cost-estimator/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore              # Git ignore file
└── .streamlit/             # Streamlit configuration (create if needed)
    └── config.toml         # Streamlit config (optional)
```

## Requirements

- Python 3.8+
- Groq API key
- Internet connection (for model downloads and API calls)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.

