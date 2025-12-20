import time
import mlx.core as mx
from PIL import Image
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

def main():
    start = time.perf_counter()
    model = build_sam3_image_model()

    second = time.perf_counter()
    print(f"Model loaded in {second - start:.2f} seconds.")
    
    image_path = f"assets/images/test_image.jpg"
    image = Image.open(image_path)
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    # mx.eval(inference_state)
    inter = time.perf_counter()
    print(f"Image processed in {inter - second:.2f} seconds.")

    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="shoe")
    output = inference_state
    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    third = time.perf_counter()
    print(f"Inference completed in {third - second:.2f} seconds.")
    print(f"Total Objects Found: {len(scores)}")

    
    # box_input_xywh = mx.array([480.0, 290.0, 110.0, 360.0]).reshape(-1, 4)
    # box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

    # norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    # print("Normalized box input:", norm_box_cxcywh)

    # processor.reset_all_prompts(inference_state)
    # inference_state = processor.add_geometric_prompt(
    #     state=inference_state, box=norm_box_cxcywh, label=True
    # )
    # output = inference_state
    # masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(scores)
    print(boxes)

    

if __name__ == "__main__":
    main()
