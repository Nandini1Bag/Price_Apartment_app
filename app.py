from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from domain.domain import ApartmentRequest, ApartmentResponse
from service.apartment_service import ApartmentService

# Initialize FastAPI app
price_app = FastAPI()

# Set up Jinja2 template engine
templates = Jinja2Templates(directory="templates")

# Mount static files (CSS, JS, images)
price_app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the HTML template
@price_app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST endpoint to predict the apartment price
@price_app.post("/predict")
async def predict_price(request: ApartmentRequest) -> ApartmentResponse:
    # Call the prediction service and return the response
    return ApartmentService().predict_price(request=request)
