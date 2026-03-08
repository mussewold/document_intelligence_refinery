import numpy as np
from doclayout_yolo import YOLOv10
from collections import Counter
from huggingface_hub import hf_hub_download
import asyncio

_model = None


def get_model():
    global _model
    if _model is None:
        weights_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        _model = YOLOv10(weights_path)
    return _model


async def detect_layout_for_page(page_np: np.ndarray):
    loop = asyncio.get_running_loop()
    model = get_model()
    results = await loop.run_in_executor(None, lambda: model.predict(page_np, imgsz=1024, conf=0.25))
    result = results[0]

    counts = {"text": 0, "table": 0, "figure": 0}
    for box in result.boxes:
        label = result.names[int(box.cls)]
        if label in counts:
            counts[label] += 1
    # Heuristic for layout complexity
    if counts["table"] > 3:
        complexity = "table_heavy"
    elif counts["figure"] > 2:
        complexity = "figure_heavy"
    elif counts["text"] > 0 and counts["table"] == 0 and counts["figure"] == 0:
        widths = [
            box.xyxy[0][2] - box.xyxy[0][0]
            for box in result.boxes
            if result.names[int(box.cls)] == "text"
        ]
        avg_width = np.mean(widths) if widths else 0
        if avg_width > page_np.shape[1] * 0.8:
            complexity = "single_column"
        else:
            complexity = "multi_column"
    else:
        complexity = "mixed"

    return complexity, counts


async def detect_layout_complexity(artifacts):
    images = await artifacts.load_images()
    tasks = [detect_layout_for_page(img) for img in images]
    results = await asyncio.gather(*tasks)

    complexity_scores = [res[0] for res in results]
    total_counts = Counter()
    for _, counts in results:
        total_counts.update(counts)
    total_counts["total_pages"] = len(images)

    final_complexity = Counter(complexity_scores).most_common(1)[0][0]
    return final_complexity, dict(total_counts)