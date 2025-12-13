from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
    )
    
    processor = Sam3Processor(model)
    # Load an image
    image_path="test_img.jpg"
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    # Prompt the model with text
    prompt = "lemons"
    output = processor.set_text_prompt(
    state=inference_state, prompt=prompt
    )

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    breakpoint()

    

if __name__ == "__main__":
    main()
