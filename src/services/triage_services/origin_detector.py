from pypdf import PdfReader


async def detect_pdf_origin(artifacts,
                         char_density_threshold=1e-5,
                         image_ratio_threshold=0.6):

    try:
        reader = PdfReader(artifacts.pdf_path)
        if reader.get_fields():
            return "form_fillable"
    except Exception:
        pass

    total_pages = 0
    digital_pages = 0
    scanned_pages = 0

    pdf = await artifacts.load_pdf()
    for page in pdf.pages:
        total_pages += 1

        page_area = page.width * page.height
        num_chars = len(page.chars)
        char_density = num_chars / page_area if page_area else 0

        image_area = sum(
            img["width"] * img["height"] for img in page.images
        )
        image_ratio = image_area / page_area if page_area else 0

        has_text = num_chars > 0
        large_image = image_ratio > image_ratio_threshold

        if char_density > char_density_threshold and image_ratio < 0.4:
            digital_pages += 1
        elif not has_text and large_image:
            scanned_pages += 1
        else:
            digital_pages += 0.5
            scanned_pages += 0.5

    if digital_pages == total_pages:
        return "native_digital"

    if scanned_pages == total_pages:
        return "scanned_image"

    return "mixed"