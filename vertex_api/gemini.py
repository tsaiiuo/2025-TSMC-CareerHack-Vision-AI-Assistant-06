#https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-pro")
print(model.generate_content("Why is sky blue?"))