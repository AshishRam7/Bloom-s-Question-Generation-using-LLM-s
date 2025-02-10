import fitz
import io
from PIL import Image
import moondream as md
from typing import List, Dict, Union
import logging
from pathlib import Path

class ScientificInsightAnalyzer:
    def __init__(self, api_key: str):
        self.logger = logging.getLogger("ScientificInsightAnalyzer")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.model = md.vl(api_key=api_key)
    
    def analyze_image_insights(self, image: Image.Image) -> str:
        try:
            encoded_image = self.model.encode_image(image)
            return self.model.query(encoded_image,
                "Describe the key technical findings in this visualization using natural language. Focus on trends, patterns, and numerical values. Provide a single paragraph summary.")["answer"]
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Union[Image.Image, int]]]:
        extracted_images = []
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    try:
                        base_image = pdf_document.extract_image(img_info[0])
                        image = Image.open(io.BytesIO(base_image["image"]))
                        extracted_images.append({"image": image})
                    except Exception as e:
                        continue
        return extracted_images
    
    def process_pdf(self, pdf_path: str) -> str:
        insights = []
        for img_data in self.extract_images_from_pdf(pdf_path):
            try:
                insight = self.analyze_image_insights(img_data["image"])
                if insight:
                    insights.append(insight)
            except Exception:
                continue
        # Join all insights without any spacing
        return ''.join(insights)

if __name__ == "__main__":
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJlODYyMDEzZC00NmVkLTRiNDYtOGMxZi0xYzYwMTUzY2M0YjkiLCJpYXQiOjE3Mzc1MjYyMjd9.0agZ8vgxwgrUJ7YMrIoBqGPs_4hsuh2zhqkwckxYkIM"
    analyzer = ScientificInsightAnalyzer(api_key)
    pdf_path = "DeepSeek_R1.pdf"
    
    try:
        insights = analyzer.process_pdf(pdf_path)
        print(insights)
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")