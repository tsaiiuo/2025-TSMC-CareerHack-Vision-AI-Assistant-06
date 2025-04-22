import io

def localize_objects_uri(uri=""):
    """Localize objects in the image on Google Cloud Storage
    Args:
    uri: The path to the file in Google Cloud Storage (gs://...)
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    # image.source.image_uri = uri
    with io.open("output_img/input-image2.png", 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    
    objects = client.object_localization(image=image).localized_object_annotations
    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        print(f"\n{object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
    for vertex in object_.bounding_poly.normalized_vertices:
        print(f" - ({vertex.x}, {vertex.y})")

localize_objects_uri()