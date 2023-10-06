import pickle
from rest_framework import status
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view


# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pickle

@api_view(['POST', 'GET'])
def predict_sneakers(request):
    if request.method == "POST":
        try:
            
            review_text = request.data.get('review_text')
            rating = request.data.get('rating')

            
            with open('mlscript/rf_model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)

            
            data = [review_text, rating]  
            prediction = model.predict([data])

            # Return the prediction as a response
            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'message': 'sneaker prediction API is working!'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

        
        
    
   
   
    
    
    