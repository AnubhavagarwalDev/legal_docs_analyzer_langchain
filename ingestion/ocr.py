import pytesseract
from PIL import Image
from pdf2image import convert_from_path



def ocr_pdf(pdf_path, dpi=300, lang='eng'):
    """
    Performs OCR on a PDF document using Tesseract.
    Returns extracted text and confidence scores for each page.
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return [], []

    all_texts = []
    all_avg_confs = []

    for i, image in enumerate(images):
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=lang)
        page_text = pytesseract.image_to_string(image, lang=lang)
        all_texts.append(page_text)

        conf_scores_numeric = []
        for conf_item in data["conf"]:
            try:
                conf_val = int(conf_item)
                if conf_val >= 0:
                    conf_scores_numeric.append(conf_val)
            except ValueError:
                pass

        if conf_scores_numeric:
            avg_conf = sum(conf_scores_numeric) / len(conf_scores_numeric)
            all_avg_confs.append(avg_conf)
        else:
            all_avg_confs.append(0.0)

    return all_texts, all_avg_confs