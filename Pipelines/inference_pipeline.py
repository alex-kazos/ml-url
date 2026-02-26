#TOD0: create a pipeline for so the user inputs a URL and the model predicts if it's phishing or not, including data loading, preprocessing, and model training steps.


from Services.preprocess_data import preprocess_data_inference_service
from Services.feature_engineering import feature_engineering_service


if __name__ == "__main__":

    input_url = input("Enter a URL to check if it's phishing or not: ")

    # Step 1: Preprocess data
    url_df = preprocess_data_inference_service(input_url)

    # Step 2: Feature engineering
    ml_ready_df = feature_engineering_service(df=url_df)

    # TODO: Step 3: Load model and predict (to be implemented)
    # model = load_model()  # Implement this function to load your trained model
    # prediction = model.predict(ml_ready_df)
    # print(f"The URL '{input_url}' is predicted to be: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
