from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predictor import run_predictions

app = FastAPI()

# Allow requests from Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
async def predict():
    try:
        run_predictions()
        return {"message": "✅ Predictions generated and saved to /public/images."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
