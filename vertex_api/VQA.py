import vertexai
from vertexai.preview.vision_models import Image, ImageTextModel

# TODO(developer): Update and un-comment below lines
# PROJECT_ID = "your-project-id"
# input_file = "input-image.png"
# question = "" # The question about the contents of the image.

PROJECT_ID = "tsmccareerhack2025-aaid-test"
input_file = "output_img/ref.png"
question = "What's in the picture?"

vertexai.init(project=PROJECT_ID, location="us-central1")

model = ImageTextModel.from_pretrained("imagetext@001")
source_img = Image.load_from_file(location=input_file)

answers = model.ask_question(
    image=source_img,
    question=question,
    # Optional parameters
    number_of_results=1,
)

print(answers)
# Example response:
# ['tabby']