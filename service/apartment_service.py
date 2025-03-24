import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from domain.domain import ApartmentRequest, ApartmentResponse
from sklearn.ensemble import RandomForestRegressor

class ApartmentService():
    def __init__(self):
        self.path_model = "artifacts/randomForestForApartmentPrice.pkl"
        self.path_encoder = "artifacts/neighbourhood_encoder.pkl"
        self.model = self.load_artifact(self.path_model)
        self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self, path_to_artifact):
        '''Load from a pickle file.'''
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact

    def preprocess_input(self, request: ApartmentRequest) -> pd.DataFrame:
        # Create a dictionary with the input data
        data_dict = {
            "rooms": request.rooms,
            "size": request.size,
            "bathrooms": request.bathrooms,
            "neighbourhood": request.neighbourhood,
            "year_built": request.year_built
        }
        
        # Convert to DataFrame
        data_df = pd.DataFrame.from_dict([data_dict])

        # Preprocess the 'neighbourhood' column (e.g., encoding it)
        data_df['neighbourhood'] = data_df['neighbourhood'].str.lower()
        data_df['neighbourhood'] = self.le.transform(data_df['neighbourhood'])
        data_df['neighbourhood'] = data_df['neighbourhood'].astype('category')

        return data_df

    def predict_price(self, request: ApartmentRequest) -> ApartmentResponse:
        # Preprocess the input data
        input_df = self.preprocess_input(request)
        
        # Make the prediction using the trained model
        apartment_price = self.model.predict(input_df)[0]
        apartment_price = int(apartment_price)  # Convert to integer for easier handling
        
        # Create an ApartmentResponse instance and return the predicted price
        response = ApartmentResponse(price=apartment_price)
        return response
